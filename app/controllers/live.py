"""
live.py — dynamic hybrid live-sentiment demo (see /live).

Deliberately stateless: no Job, no FileRun, no SQLite. The point of this page is response time,
so nothing here touches the database or the async job pipeline that the rest of the app uses.
The browser scores locally once it decides the device can keep up (see live_sentiment.js); this
endpoint is the same algorithm run server-side, for the remote-mode fallback and for the initial
render before the client has decided which mode to use.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.deps import get_registry, get_settings
from app.core.templating import render

router = APIRouter()


class LiveIn(BaseModel):
    text: str = Field("", max_length=2000)


@router.get("/live", response_class=HTMLResponse)
def page(request: Request, settings=Depends(get_settings), registry=Depends(get_registry)):
    prof = settings.profile("live_sentiment")
    backend = registry.get("lexicon_sentiment")  # loaded now, not on first keystroke
    return render(request, "live_sentiment.html", lexicon_url=f"/static/{prof.lexicon_asset}",
                  neutral_band=prof.neutral_band, negation_window=backend.window,
                  lexicon_kb=round(backend.ASSET_PATH.stat().st_size / 1024, 1))


@router.post("/api/live/sentiment")
def score(body: LiveIn, registry=Depends(get_registry)):
    """The server-side half of the hybrid. Same algorithm, same lexicon file, as the browser port."""
    backend = registry.get("lexicon_sentiment")
    avg, matches = backend.score(body.text)
    matches.sort(key=lambda m: -abs(m["contribution"] * m["weight"]))
    signals = backend.signal_summary(matches)
    return {"label": backend.label_for(avg), "score": avg, "confidence": backend.confidence_for(avg, matches),
            "matches": matches[:12], "mode": "server", **signals}
