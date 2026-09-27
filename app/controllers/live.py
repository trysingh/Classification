"""
live.py — dynamic hybrid live-sentiment demo (see /live).

Deliberately stateless: no Job, no FileRun, no SQLite. The point of this page is response time,
so nothing here touches the database or the async job pipeline that the rest of the app uses.
The browser scores locally once it decides the device can keep up (see live_sentiment.js); this
endpoint is the same algorithm run server-side, for the remote-mode fallback and for the initial
render before the client has decided which mode to use.
"""
import threading

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.deps import get_registry, get_settings
from app.core.errors import BackendError
from app.core.templating import render

router = APIRouter()
# A real System-1 model call can take seconds; this caps escalation at ONE in flight at a time,
# globally, regardless of how many requests arrive (multiple tabs, a client debounce bug, etc.)
# -- extra requests get the lexicon result immediately rather than queueing more expensive work
# behind an already-running call. The client-side debounce (live_sentiment.js) is the primary
# fix for call volume; this is the backstop that holds even if that debounce is ever bypassed.
_escalation_lock = threading.Lock()


class LiveIn(BaseModel):
    text: str = Field("", max_length=2000)
    backend: str | None = None       # per-request override of the escalation model (debug/eval)
    force_system1: bool = False      # ask the model even when the lexicon DID find matches (debug/eval)


@router.get("/live", response_class=HTMLResponse)
def page(request: Request, settings=Depends(get_settings), registry=Depends(get_registry)):
    prof = settings.profile("live_sentiment")
    backend = registry.get("lexicon_sentiment")  # loaded now, not on first keystroke
    return render(request, "live_sentiment.html", lexicon_url=f"/static/{prof.lexicon_asset}",
                  neutral_band=prof.neutral_band, negation_window=backend.window,
                  lexicon_kb=round(backend.ASSET_PATH.stat().st_size / 1024, 1))


@router.post("/api/live/sentiment")
def score(body: LiveIn, settings=Depends(get_settings), registry=Depends(get_registry)):
    """The server-side half of the hybrid. Same lexicon algorithm as the browser port when it
    has any vocabulary to go on. When it finds NOTHING at all -- as opposed to a low-but-nonzero
    score, which is a legitimate "mostly neutral" reading -- a flat Neutral would be misleading:
    it isn't that the text IS neutral, it's that the lexicon has no data for it (e.g. "sounds
    soothing to soul" has no word this ~150-entry lexicon knows). In that specific case only,
    escalate to a real System-1 model instead of guessing. A model that fails to load degrades
    quietly back to the honest lexicon result -- this is inline, live-typing UX, not a batch job,
    so it should never interrupt someone mid-sentence with an error box."""
    lex = registry.get("lexicon_sentiment")
    text = body.text.strip()
    avg, matches = lex.score(text) if text else (0.0, [])
    matches.sort(key=lambda m: -abs(m["contribution"] * m["weight"]))
    signals = lex.signal_summary(matches)
    base = {"label": lex.label_for(avg), "score": avg, "confidence": lex.confidence_for(avg, matches),
            "matches": matches[:12], "mode": "server", "source": "lexicon", **signals}

    prof = settings.profile("live_sentiment")
    should_escalate = text and (body.force_system1 or not matches) and (prof.escalation_backend or body.backend)
    if not should_escalate:
        return base

    if not _escalation_lock.acquire(blocking=False):
        return base | {"escalation_error": "A previous request to the model is still being processed; showing the lexicon result for now."}
    try:
        backend_name = settings.backend_for("live_sentiment", override=body.backend, field="escalation_backend")
        model = registry.get(backend_name)
        labels = list(prof.seed_taxonomy) or ["Positive", "Negative", "Neutral"]
        question = settings.prompt("live_sentiment", "main_category_question")
        picked = model.choose(text, question, labels, debias=1)   # debias=1: this is interactive, not a batch run
    except BackendError as e:
        return base | {"escalation_error": f"{e.message} {e.hint or ''}".strip()}
    finally:
        _escalation_lock.release()

    top_label = picked["choice"]
    top_conf = round(picked["probabilities"].get(top_label, 0.0), 4)
    pseudo_score = {"Positive": top_conf, "Negative": -top_conf}.get(top_label, 0.0) * 3   # same -3..3 bar scale as the lexicon
    return base | {"label": top_label, "score": round(pseudo_score, 4), "confidence": top_conf,
                   "source": "system1", "backend": backend_name, "probabilities": picked["probabilities"]}
