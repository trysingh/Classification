"""
templating.py
-------------
Jinja2 environment shared by all controllers: display filters (dates, numbers, sizes, stable
category colours) and a `render()` helper so every page gets the same base context.
Formatting lives here, not in controllers or templates, so wording/units change in one place.
"""
from __future__ import annotations

import zlib
from datetime import datetime
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

STATUS_LABELS = {"queued": "Queued", "running": "Running", "completed": "Completed",
                 "completed_with_errors": "Completed with errors", "failed": "Failed", "cancelled": "Cancelled"}
BACKEND_LABELS = {"causal_lm": "Qwen logit read-off", "laya": "Laya", "keyword": "Keyword baseline"}


def fmt_dt(v) -> str:
    return v.strftime("%d %b %Y, %H:%M") if isinstance(v, datetime) else "-"


def fmt_num(v, digits: int = 0) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def fmt_pct(v, digits: int = 0) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def fmt_dur(sec) -> str:
    if sec is None:
        return "-"
    if sec < 1:
        return "<1s"
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def fmt_size(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def cat_class(name: str) -> str:
    """Stable colour per category name, so a category looks the same on every page."""
    if name == templates.env.globals.get("others_label"):
        return "c-others"
    return f"c{zlib.crc32(str(name).encode()) % 8}"


templates.env.filters.update(dt=fmt_dt, num=fmt_num, pct=fmt_pct, dur=fmt_dur, size=fmt_size, cat_class=cat_class,
                             status_label=lambda s: STATUS_LABELS.get(s, s),
                             backend_label=lambda b: BACKEND_LABELS.get(b, b))


def render(request: Request, name: str, status_code: int = 200, **ctx):
    """Render a template with the flash messages carried in the query string (?ok=... / ?err=...)."""
    ctx.setdefault("notice", request.query_params.get("ok"))
    ctx.setdefault("notice_error", request.query_params.get("err"))
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)
