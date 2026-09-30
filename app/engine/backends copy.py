"""
backends.py
-----------
System-1 back-ends behind ONE tiny contract, so the hierarchical algorithm (hierarchical.py)
is written once and any model can be swapped in via config:

    choose(context, question, labels) -> {"choice": str, "probabilities": {label: p}}
    generate_label(narration, prompt) -> short free-text label (only used for the 'Others' branch)

Heavy libraries are imported lazily, so the app boots (and the keyword baseline works)
without torch / transformers / laya installed. EngineRegistry loads each model once and
shares it across all jobs.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import random
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.errors import BackendError

log = logging.getLogger(__name__)


class ChoiceBackend(ABC):
    name = "base"

    @abstractmethod
    def choose(self, context: str, question: str, labels: list[str], *,
               debias: int = 1, hints: dict[str, str] | None = None) -> dict:
        """Pick one label. `hints` = optional extra text per label (only the keyword baseline uses it)."""

    @abstractmethod
    def generate_label(self, narration: str, prompt: str) -> str:
        """Name a specific sub-category when a row lands in 'Others'."""


class CausalLMBackend(ChoiceBackend):
    """Qwen-style causal LM: reads the logits of the option letters (A, B, C...) after 'Answer:'.
    Option order is shuffled `debias` times and averaged per label to cancel positional bias."""

    name = "causal_lm"

    def __init__(self, settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.system1_model, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(settings.system1_model, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()          # one forward pass at a time; single + batch workers share the model
        log.info("Loaded causal LM %s on %s (%s)", settings.system1_model, self.device, dtype)

    @staticmethod
    def _prompt(context: str, question: str, labels: list[str]) -> str:
        options = "\n".join(f"{chr(65 + i)}. {label}" for i, label in enumerate(labels))
        return (f"Context: {context}\nQuestion: {question}\nOptions:\n{options}\n"
                f"Answer with only the letter of the best option.\nAnswer:")

    def _single_pass(self, context: str, question: str, labels: list[str]) -> dict[str, float]:
        torch = self._torch
        inputs = self.tokenizer(self._prompt(context, question, labels), return_tensors="pt").to(self.device)
        ids = [self.tokenizer.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(len(labels))]
        with self._lock, torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]          # single forward pass
        probs = torch.softmax(logits[ids], dim=-1)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}

    def choose(self, context, question, labels, *, debias=1, hints=None) -> dict:
        n = max(1, debias)
        rng = random.Random(42)                                  # fixed seed: reproducible, still varies the order
        agg = {label: 0.0 for label in labels}
        for _ in range(n):
            order = labels[:]
            rng.shuffle(order)
            for label, p in self._single_pass(context, question, order).items():
                agg[label] += p
        agg = {label: round(p / n, 4) for label, p in agg.items()}
        return {"choice": max(agg, key=agg.get), "probabilities": agg}

    def generate_label(self, narration: str, prompt: str) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._lock, torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=6, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0].strip(" .") or "Unclassified"


class LayaBackend(ChoiceBackend):
    """Convai Laya: non-autoregressive option scoring, immune to letter-position bias (no permutations).
    Laya cannot generate text, so the 'Others' label borrows the causal LM (loaded only if needed)."""

    name = "laya"

    def __init__(self, settings, registry: "EngineRegistry"):
        import laya as laya_lib

        self.agent = laya_lib.load(settings.laya_model_name)
        self._registry = registry
        log.info("Loaded Laya model %s", settings.laya_model_name)

    def choose(self, context, question, labels, *, debias=1, hints=None) -> dict:
        q = {"q": {"type": "choice", "instructions": question, "criteria": {label: label for label in labels}}}
        ans = self.agent.predict({"body": context}, q)["answers"]["q"]
        choice = ans["choice"]
        # The per-option field name isn't pinned down by the model card: try the likely keys,
        # else fall back to chosen + its confidence.
        probs = ans.get("probabilities") or ans.get("distribution") or ans.get("scores")
        probs = {k: float(v) for k, v in probs.items()} if probs else {choice: float(ans.get("confidence", 0.0))}
        return {"choice": choice, "probabilities": probs}

    def generate_label(self, narration: str, prompt: str) -> str:
        return self._registry.get("causal_lm").generate_label(narration, prompt)


_STOP = {"the", "and", "for", "of", "to", "in", "on", "an", "with", "by", "at", "from", "is", "are", "as",
         "or", "per", "via", "this", "that", "be", "it", "its", "our", "your", "new"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


class KeywordBackend(ChoiceBackend):
    """Dependency-free baseline: word overlap between the text and each label (plus its sub-labels for
    level 1). Not a substitute for the models: it exists for smoke tests, offline demos and CI."""

    name = "keyword"

    def __init__(self, settings):
        self.others = settings.others_label

    def choose(self, context, question, labels, *, debias=1, hints=None) -> dict:
        ctx = set(_tokens(context))
        scores = {}
        for label in labels:
            own = set(_tokens(label))
            extra = set(_tokens((hints or {}).get(label, ""))) - own
            scores[label] = 2.0 * len(own & ctx) + 1.0 * len(extra & ctx)
        if max(scores.values()) <= 0:                            # no signal: say so honestly (uniform, low confidence)
            choice = self.others if self.others in labels else labels[0]
            return {"choice": choice, "probabilities": {label: round(1 / len(labels), 4) for label in labels}}
        exps = {label: math.exp(s) for label, s in scores.items()}
        z = sum(exps.values())
        probs = {label: round(v / z, 4) for label, v in exps.items()}
        return {"choice": max(probs, key=probs.get), "probabilities": probs}

    def generate_label(self, narration: str, prompt: str) -> str:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z&-]{3,}", narration) if w.lower() not in _STOP]
        return " ".join(dict.fromkeys(w.title() for w in words[:3])) or "Unclassified"


class LexiconSentimentBackend(ChoiceBackend):
    """Valence-lexicon scorer for the live-sentiment demo (see /live): no model, no network,
    still sub-millisecond even with v2's extra passes. static/js/live_sentiment.js is a
    line-for-line port of the methods below, reading the SAME lexicon file
    (app/static/data/live_sentiment_lexicon.json), so browser-side and server-side scoring
    never disagree -- only which one runs for a given keystroke changes.

    v2 adds four things a single word-by-word pass can't do:
      - phrases: literal multi-word overrides ("not bad", "waste of money"...), matched greedily
        (longest first) before word-level scoring, so a phrase's own words aren't ALSO scored
        individually and double-counted.
      - clause splitting: text is cut first on sentence punctuation, then within each sentence on
        a contrast word (but/however/...). This is what actually fixes negation windows bleeding
        across sentences, and lets the clause AFTER a contrast word outweigh the one before it
        ("great UI but slow app" leans negative, not a wash).
      - diminishers, split out from intensifiers (values <1 moved here) so the two concepts don't
        share one dict.
      - emojis and a capped CAPS/punctuation emphasis multiplier, both computed from the raw text
        rather than the tokenizer (which only ever sees `[a-z']+`).

    score()/label_for()/confidence_for() are the direct API the /live demo endpoint calls.
    choose() adapts the same scoring to the generic ChoiceBackend contract, so this backend also
    works unmodified through HierarchicalClassifier for the ordinary batch/single-file pipeline.
    """

    name = "lexicon_sentiment"
    ASSET_PATH = Path(__file__).resolve().parent.parent / "static" / "data" / "live_sentiment_lexicon.json"
    _TOKEN = re.compile(r"[a-z']+")
    _SENTENCE_SPLIT = re.compile(r"[.!?]+")
    _CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")
    _EXCLAIM = re.compile(r"[!?]{1,}$")

    def __init__(self, settings):
        try:
            data = json.loads(self.ASSET_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise BackendError("The sentiment lexicon file is missing or invalid.",
                               hint=f"Check {self.ASSET_PATH}.", detail=str(e))
        self.words: dict[str, float] = data["words"]
        # Cheap coverage extension: "crashing"/"crashes" hit the same entry as "crash" without
        # listing every inflection by hand. Reuses the suffix-stripper KeywordBackend already
        # uses elsewhere in this file -- same tradeoff (single pass, no recursion), not a real
        # stemmer. First word to reach a given stem wins if two collide.
        self._stem_index: dict[str, float] = {}
        for w, s in self.words.items():
            self._stem_index.setdefault(_stem(w), s)
        self.phrases: list[tuple[list[str], float]] = sorted(
            ((p.split(), s) for p, s in data.get("phrases", {}).items()), key=lambda ps: -len(ps[0]))
        self.negators: set[str] = set(data["negators"])
        self.intensifiers: dict[str, float] = data["intensifiers"]
        self.diminishers: dict[str, float] = data.get("diminishers", {})
        self.emojis: dict[str, float] = data.get("emojis", {})
        meta = data.get("meta", {})
        self.window: int = meta.get("negation_window", 3)
        self.scope_breakers: set[str] = set(meta.get("scope_breakers", []))
        self.cap_boost_max: float = meta.get("cap_boost_max", 1.3)
        self.punct_boost_max: float = meta.get("punct_boost_max", 1.3)
        prof = settings.profiles.get("live_sentiment")
        self.neutral_band = prof.neutral_band if prof else 0.15
        log.info("Loaded sentiment lexicon v%s: %d words, %d phrases, %d emojis",
                 meta.get("version", "?"), len(self.words), len(self.phrases), len(self.emojis))

    def tokenize(self, text: str) -> list[str]:
        return self._TOKEN.findall(text.lower())

    def _emphasis_multiplier(self, text: str) -> float:
        """CAPS and repeated ! / ? give a small boost, but the COMBINED total is what's capped --
        capping each source separately still let them stack past the intended ceiling, which
        would have let someone inflate a score just by shouting AND mashing punctuation at once."""
        words_all = text.split()
        caps = len(self._CAPS_WORD.findall(text))
        caps_ratio = caps / len(words_all) if words_all else 0.0
        caps_boost = caps_ratio * 0.6
        bangs = self._EXCLAIM.search(text.strip())
        punct_boost = 0.08 * len(bangs.group()) if bangs else 0.0
        overall_max = max(self.cap_boost_max, self.punct_boost_max) - 1.0
        return round(1.0 + min(caps_boost + punct_boost, overall_max), 4)

    def _emoji_matches(self, text: str) -> list[dict]:
        matches = []
        for emo, base in self.emojis.items():
            count = text.count(emo)
            if count:
                # diminishing returns for repetition (matches the JS port): 1x, then +0.2 per repeat, capped at 1.5x
                mult = min(1.0 + 0.2 * (count - 1), 1.5)
                matches.append({"type": "emoji", "word": emo, "base": base, "multiplier": round(mult, 3),
                                "negated": False, "contribution": round(base * mult, 3), "weight": 1.0})
        return matches

    def _match_phrases(self, tokens: list[str]) -> tuple[list[dict], set[int]]:
        consumed: set[int] = set()
        matches: list[dict] = []
        for words, score in self.phrases:                    # longest phrases first (see __init__ sort)
            n = len(words)
            i = 0
            while i <= len(tokens) - n:
                if all(idx not in consumed for idx in range(i, i + n)) and tokens[i:i + n] == words:
                    matches.append({"type": "phrase", "word": " ".join(words), "base": score, "multiplier": 1.0,
                                    "negated": False, "contribution": round(score, 3)})
                    consumed.update(range(i, i + n))
                    i += n
                else:
                    i += 1
        return matches, consumed

    def _score_tokens(self, tokens: list[str], weight: float) -> list[dict]:
        phrase_matches, consumed = self._match_phrases(tokens)
        for m in phrase_matches:
            m["weight"] = weight
        word_matches = []
        for i, tok in enumerate(tokens):
            if i in consumed:
                continue
            base = self.words.get(tok)
            if base is None:
                base = self._stem_index.get(_stem(tok))
            if base is None:
                continue
            mult, negate = 1.0, False
            for back in range(1, self.window + 1):
                j = i - back
                if j < 0:
                    break
                t2 = tokens[j]
                negate = negate or t2 in self.negators
                if t2 in self.intensifiers:
                    mult = self.intensifiers[t2]
                elif t2 in self.diminishers:
                    mult = self.diminishers[t2]
            val = round(base * mult * (-1.0 if negate else 1.0), 3)
            word_matches.append({"type": "word", "word": tok, "base": base, "multiplier": mult,
                                 "negated": negate, "contribution": val, "weight": weight})
        return phrase_matches + word_matches

    def _score_sentence(self, sentence: str) -> list[dict]:
        tokens = self.tokenize(sentence)
        breaker_idx = next((i for i, t in enumerate(tokens) if t in self.scope_breakers), None)
        if breaker_idx is None:
            return self._score_tokens(tokens, weight=1.0)
        # The clause AFTER a contrast word ("but", "however"...) usually carries the speaker's real
        # point ("great UI but slow app"), so it's weighted up; the clause before is weighted down.
        # This also incidentally stops a negator before the breaker from reaching past it.
        before = self._score_tokens(tokens[:breaker_idx], weight=0.6)
        after = self._score_tokens(tokens[breaker_idx + 1:], weight=1.6)
        return before + after

    def score(self, text: str) -> tuple[float, list[dict]]:
        """Stateless (only reads immutable data set in __init__), safe under concurrent calls.
        Returns (weighted-average score, matches) -- matches carry 'type' (word/phrase/emoji) and
        'weight' (clause weighting) for callers that want more than the headline number."""
        matches: list[dict] = []
        for sentence in self._SENTENCE_SPLIT.split(text):
            if sentence.strip():
                matches += self._score_sentence(sentence)
        matches += self._emoji_matches(text)
        if not matches:
            return 0.0, []
        weight_total = sum(m["weight"] for m in matches) or 1.0
        raw_avg = sum(m["contribution"] * m["weight"] for m in matches) / weight_total
        avg = round(raw_avg * self._emphasis_multiplier(text), 4)
        return avg, matches

    def signal_summary(self, matches: list[dict]) -> dict:
        """positive_score/negative_score/mixed: kept separate from the single label so a caller
        that wants more than Positive/Negative/Neutral (e.g. flagging 'mixed signals' in the UI)
        doesn't have to re-derive it from raw matches."""
        pos = round(sum(m["contribution"] * m["weight"] for m in matches if m["contribution"] > 0), 4)
        neg = round(sum(m["contribution"] * m["weight"] for m in matches if m["contribution"] < 0), 4)
        return {"positive_score": pos, "negative_score": neg, "mixed": pos > 0.5 and neg < -0.5}

    def label_for(self, avg: float) -> str:
        if avg > self.neutral_band:
            return "Positive"
        if avg < -self.neutral_band:
            return "Negative"
        return "Neutral"

    def confidence_for(self, avg: float, matches: list[dict] | None = None) -> float:
        """More than magnitude alone: also rewards having several agreeing signals, and pulls
        back when the text is mixed (positive and negative both present in real strength)."""
        magnitude = min(abs(avg) / ((self.neutral_band * 4) or 1e-9), 1.0)
        if not matches:
            return round(magnitude, 4)
        volume = min(len(matches) / 4, 1.0)
        agreement = 0.5 if self.signal_summary(matches)["mixed"] else 1.0
        return round(min(0.6 * magnitude + 0.25 * volume * agreement + 0.15 * agreement, 1.0), 4)

    @staticmethod
    def _bucket(labels: list[str]) -> dict[str, str]:
        """Best-effort map from whatever label names the taxonomy actually has to pos/neg/neutral,
        so a renamed or extended taxonomy still gets a sensible choice rather than an error."""
        out: dict[str, str] = {}
        for label in labels:
            low = label.lower()
            if "pos" in low:
                out.setdefault("pos", label)
            elif "neg" in low:
                out.setdefault("neg", label)
            else:
                out.setdefault("neu", label)
        return out

    def choose(self, context, question, labels, *, debias=1, hints=None) -> dict:
        avg, matches = self.score(context)
        bucket = self._bucket(labels)
        choice = bucket.get({"Positive": "pos", "Negative": "neg", "Neutral": "neu"}[self.label_for(avg)]) or labels[0]
        conf = self.confidence_for(avg, matches)
        rest = (1.0 - conf) / max(1, len(labels) - 1)
        probs = {label: (conf if label == choice else rest) for label in labels}
        total = sum(probs.values()) or 1.0
        return {"choice": choice, "probabilities": {label: round(p / total, 4) for label, p in probs.items()}}

    def generate_label(self, narration: str, prompt: str) -> str:
        avg, matches = self.score(narration)
        if not matches:
            return "No sentiment words found"
        top = max(matches, key=lambda m: abs(m["contribution"]))
        return f"Driven by '{top['word']}'"


class ReflexDecisionBackend(ChoiceBackend):
    """Deterministic, sub-millisecond action chooser for real-time twitch games (see /decide).
    Same shared-asset pattern as LexiconSentimentBackend: this class and the mode's JS adapter
    (static/js/dino_adapter.js etc.) read the SAME rules JSON
    (app/static/data/<mode>_reflex_rules.json by convention, or the profile's reflex_asset if
    set), so browser-side control and this server-side endpoint never disagree -- only which
    one drives the actual keypresses differs.

    Unlike LexiconSentimentBackend this backend is state-driven, not text-driven: choose()'s
    `context` is a JSON-encoded game state, and `labels` is the mode's action set. No text is
    actually read for the decision -- context/labels are accepted for ChoiceBackend /
    HierarchicalClassifier interface compatibility; the real per-tick entry point is decide().

    decide() deliberately takes no `calibration` argument, unlike its JS twin: the client-side
    lead_multiplier nudging in decision_engine.js's calibrate() is ephemeral and browser-local
    by design (see that file's docstring), so a server review always answers against the un-
    nudged baseline rules file, never a copy that's drifted from what's on disk.
    """

    name = "reflex_decision"
    DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "data"

    def __init__(self, settings, mode: str = "dino"):
        # settings.profile() raises a clear ConfigError for an unknown mode before we even get
        # to file I/O, and profile.reflex_asset -- when set -- takes precedence over the plain
        # <mode>_reflex_rules.json naming convention dino_adapter.js's fetch() also assumes.
        prof = settings.profile(mode)
        path = (Path(__file__).resolve().parent.parent / "static" / prof.reflex_asset) if prof.reflex_asset \
            else self.DATA_DIR / f"{mode}_reflex_rules.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise BackendError(f"The '{mode}' reflex rules file is missing or invalid.",
                               hint=f"Check {path}.", detail=str(e))
        self.mode = mode
        self.actions: list[str] = data["actions"]
        self.physics: dict = data.get("physics", {})
        self.obstacle_profiles: dict = data.get("obstacle_profiles", {})
        self.lead_multiplier: float = data.get("thresholds", {}).get("lead_multiplier", 1.15)
        self.ambiguous_band: float = data.get("thresholds", {}).get("ambiguous_margin_px", 12)
        log.info("Loaded reflex rules for mode=%s: %d actions, %d obstacle profiles",
                 mode, len(self.actions), len(self.obstacle_profiles))

    def _airtime_frames(self) -> float:
        g = self.physics.get("GRAVITY", 0.6)
        v = self.physics.get("INITIAL_JUMP_VELOCITY", 12)
        return 2.0 * v / g if g else 40.0

    def decide(self, state: dict) -> dict:
        """The actual reflex: identical logic to dino_adapter.js's decide() for this mode.
        state: {"obstacle_type": str|None, "obstacle_x": float, "obstacle_y": float, "speed": float}
        Returns {"action", "confidence", "ambiguous", "trigger_px"}; ambiguous=True is the
        signal the caller (decide.py) should escalate to a System-1 model via choose()."""
        obstacle_type = state.get("obstacle_type")
        if not obstacle_type:
            return {"action": "Run", "confidence": 1.0, "ambiguous": False, "trigger_px": None}
        speed = max(state.get("speed", self.physics.get("SPEED", 6)), 0.1)
        trigger_px = speed * self._airtime_frames() * self.lead_multiplier
        dist = state.get("obstacle_x", 9999)
        if dist > trigger_px + self.ambiguous_band:
            return {"action": "Run", "confidence": 1.0, "ambiguous": False, "trigger_px": trigger_px}
        profile = self.obstacle_profiles.get(obstacle_type, {})
        base_action = profile.get("action", "Jump")
        margin = trigger_px - dist
        ambiguous = abs(margin) <= self.ambiguous_band
        confidence = 0.5 if ambiguous else round(min(abs(margin) / ((self.ambiguous_band * 2) or 1e-9), 1.0), 4)
        return {"action": base_action, "confidence": confidence, "ambiguous": ambiguous, "trigger_px": trigger_px}

    def choose(self, context, question, labels, *, debias=1, hints=None) -> dict:
        try:
            state = json.loads(context)
        except (TypeError, json.JSONDecodeError):
            state = {}
        result = self.decide(state)
        action = result["action"] if result["action"] in labels else labels[0]
        conf = result["confidence"]
        rest = (1.0 - conf) / max(1, len(labels) - 1)
        probs = {label: (conf if label == action else rest) for label in labels}
        total = sum(probs.values()) or 1.0
        return {"choice": action, "probabilities": {label: round(p / total, 4) for label, p in probs.items()}}

    def generate_label(self, narration: str, prompt: str) -> str:
        return "Reflex decision (no generative label)"


class EngineRegistry:
    """Loads each backend once (lazily, thread-safe) and hands the shared instance to every job."""

    def __init__(self, settings):
        self.settings = settings
        self._cache: dict[str, ChoiceBackend] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> ChoiceBackend:
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._build(name)
            return self._cache[name]

    def _build(self, name: str) -> ChoiceBackend:
        s = self.settings
        try:
            if name == "causal_lm":
                return CausalLMBackend(s)
            if name == "laya":
                return LayaBackend(s, self)
            if name == "keyword":
                return KeywordBackend(s)
            if name == "lexicon_sentiment":
                return LexiconSentimentBackend(s)
            if name == "reflex_decision" or name.startswith("reflex_decision:"):
                mode = name.split(":", 1)[1] if ":" in name else "dino"
                return ReflexDecisionBackend(s, mode=mode)
        except ImportError as e:
            need = {"causal_lm": "pip install torch transformers", "laya": "pip install laya"}.get(name, "")
            raise BackendError(f"The '{name}' engine needs the package '{e.name}', which is not installed.",
                               hint=f"{need}. Or pick another engine (e.g. keyword) for this run.", detail=str(e))
        except BackendError:
            raise   # already specific (e.g. a missing/invalid rules or lexicon file) -- don't bury it below
        except Exception as e:  # model download / load failures
            raise BackendError(f"The '{name}' engine could not be loaded.",
                               hint="Check the model name, disk space and network access to the model hub.",
                               detail=f"{type(e).__name__}: {e}")
        raise BackendError(f"Unknown engine '{name}'.",
                           hint="Use causal_lm, laya, keyword, lexicon_sentiment or reflex_decision.")

    def availability(self) -> list[dict]:
        have = lambda m: importlib.util.find_spec(m) is not None
        rows = [("causal_lm", "Qwen logit read-off", have("torch") and have("transformers"), "torch, transformers"),
                ("laya", "Laya", have("laya"), "laya"),
                ("keyword", "Keyword baseline (no ML)", True, "nothing"),
                ("lexicon_sentiment", "Lexicon sentiment (live)", True, "nothing"),
                ("reflex_decision", "Reflex decision rules (per game mode)", True, "nothing")]
        loaded = lambda n: any(k == n or k.startswith(n + ":") for k in self._cache)
        return [{"name": n, "label": lbl, "installed": ok, "loaded": loaded(n), "needs": need}
                for n, lbl, ok, need in rows]
