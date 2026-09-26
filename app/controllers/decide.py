"""
decide.py — generic real-time "quick decision" engine (see /decide/<mode>).

Generalizes live.py's hybrid client/server pattern from text sentiment scoring to any
low-latency action-selection loop (Chrome Dino, endless runners, twitch-reaction games...).
Two speeds, same split as live_sentiment:

  - LOCAL, every tick: a deterministic reflex (ReflexDecisionBackend, see backends.py), sub-
    millisecond, no network. This is what actually drives the keypresses; it is a straight
    port of the matching JS adapter's decide(), same never-disagree guarantee the sentiment
    lexicon already gives you between browser and server.
  - REMOTE, throttled and ADVISORY ONLY: a System-1 model call (choose() on laya / causal_lm /
    keyword, picked via settings.backend_for(mode, override=...)) that only runs when the
    reflex itself flags a decision ambiguous, or the caller asks for it explicitly. It never
    blocks a game tick -- see decision_engine.js.

Deliberately stateless per request, same reasoning as live.py: nothing here touches Job / DB.
The "dino" profile in app/config.py is NOT meant to run through the batch/single-file pipeline
(HierarchicalClassifier) -- there's no text column for a per-tick game state. It exists purely
to hand this controller its action set (seed_taxonomy keys), reflex_asset, and default
System-1 escalation backend.

The page itself is rendered through the app's normal render()/base.html, not a standalone HTML
string: game_scripts + adapter_factory on the profile (see config.py) tell the template which
JS to include and which global factory function builds the {reflex, applyAction, actions}
adapter, so a second mode needs a new profile + rules JSON + adapter JS, not a template edit.
"""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.deps import get_registry, get_settings
from app.core.errors import NotFound
from app.core.templating import render

router = APIRouter()


class DecideIn(BaseModel):
    state: dict = Field(default_factory=dict)   # e.g. {"obstacle_type","obstacle_x","obstacle_y","speed"}
    backend: str | None = None                  # per-request override: "laya" | "causal_lm" | "keyword"
    force_system1: bool = False                 # ask the model even when the reflex is confident (debug/eval)


@router.get("/decide/{mode}", response_class=HTMLResponse)
def page(request: Request, mode: str, settings=Depends(get_settings), registry=Depends(get_registry)):
    if mode not in settings.profiles:
        raise NotFound(f"'{mode}' is not a known mode.", hint=f"Available: {', '.join(settings.profiles)}.")
    prof = settings.profile(mode)
    if not prof.adapter_factory:
        raise NotFound(f"'{mode}' has no playable page.",
                       hint="This profile isn't set up as a /decide mode -- set adapter_factory "
                            "and game_scripts on it in config.py.")
    registry.get(f"reflex_decision:{mode}")      # fail fast here if the reflex rules file is missing/invalid
    return render(request, "decide.html", mode=mode, prof=prof)


@router.post("/api/decide/{mode}")
def decide(mode: str, body: DecideIn, settings=Depends(get_settings), registry=Depends(get_registry)):
    """The server-side half of the hybrid. `state` is a compact game-state dict, not text --
    see ReflexDecisionBackend.decide() for its shape. Escalates to a System-1 model only when
    the reflex says ambiguous=True (or force_system1), exactly mirroring decision_engine.js's
    client-side ambiguity gate, so the "when do we bother a model" rule lives in one place per
    side rather than drifting apart."""
    if mode not in settings.profiles:
        raise NotFound(f"'{mode}' is not a known mode.", hint=f"Available: {', '.join(settings.profiles)}.")
    prof = settings.profile(mode)
    labels = list(prof.seed_taxonomy) or ["Run"]
    reflex = registry.get(f"reflex_decision:{mode}")
    result = reflex.decide(body.state)

    if not (body.force_system1 or result["ambiguous"]):
        return {"mode": mode, "action": result["action"], "confidence": result["confidence"],
                "reflex": result, "backend": "reflex_decision", "source": "reflex"}

    # backend_for() already gives per-request swap for free (profile default -> global default
    # -> this override) -- nothing new needed here to "dynamically exchange Laya for a better
    # model": change DEFAULT_PROFILES["dino"].backend, OPENJEV_CLASSIFIER_BACKEND, or pass
    # {"backend": "causal_lm"} in the request body, any of the three, no restart required
    # except for actually loading a new model the first time (EngineRegistry caches it after).
    backend_name = settings.backend_for(mode, override=body.backend)
    model = reflex if backend_name == "reflex_decision" else registry.get(backend_name)
    question = settings.prompt(mode, "main_category_question")
    picked = model.choose(json.dumps(body.state), question, labels)
    return {"mode": mode, "action": picked["choice"], "probabilities": picked["probabilities"],
            "reflex": result, "backend": backend_name, "source": "system1"}
