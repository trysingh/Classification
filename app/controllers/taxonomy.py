"""taxonomy.py — view and change the taxonomy the engine classifies against."""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_settings
from app.core.errors import AppError, NotFound, TaxonomyError, ValidationFailed
from app.core.templating import render
from app.engine import system2
from app.models.entities import FileRun, RowResult
from app.services import taxonomy_service as tax

router = APIRouter()


class SaveIn(BaseModel):
    profile: str
    taxonomy: dict[str, list[str]]
    note: str = ""


class GenerateIn(BaseModel):
    profile: str
    file_id: int
    mode: str = "merge"                  # merge = keep my categories, add new ones | replace


class RestoreIn(BaseModel):
    profile: str
    version_id: int


@router.get("/taxonomy", response_class=HTMLResponse)
def page(request: Request, profile: str = "", db: Session = Depends(get_db), settings=Depends(get_settings)):
    key = profile if profile in settings.profiles else settings.default_profile
    file_error, current = None, None
    try:
        current = tax.load_current(db, settings, key)
    except TaxonomyError as e:                                   # broken hand-edited file: show it, fall back to last good version
        file_error = f"{e.message} {e.hint or ''}"
        last = tax.latest(db, key)
        current = (json.loads(last.content_json), last.version) if last else None
    prof = settings.profile(key)
    files = db.scalars(select(FileRun).where(FileRun.profile == key, FileRun.status == "completed")
                       .order_by(FileRun.id.desc()).limit(30)).all()
    return render(request, "taxonomy.html", profile=key, prof=prof, profiles={k: p.label for k, p in settings.profiles.items()},
                  taxonomy=current[0] if current else None, version=current[1] if current else None,
                  starter=prof.seed_taxonomy or None, history=tax.history(db, key), files=files, file_error=file_error,
                  limits={"main": settings.eff(key, "max_main_categories"), "sub": settings.eff(key, "max_sub_categories")},
                  system2_enabled=settings.system2_enabled, system2_model=settings.system2_model,
                  system2_timeout=settings.system2_timeout_sec, others=settings.others_label)


@router.post("/api/taxonomy/save")
def save(body: SaveIn, db: Session = Depends(get_db), settings=Depends(get_settings)):
    settings.profile(body.profile)
    ver, warnings = tax.save(db, settings, body.profile, body.taxonomy, source="manual", note=body.note or "Edited in the app")
    return {"version": ver.version, "taxonomy": json.loads(ver.content_json), "warnings": warnings}


@router.post("/api/taxonomy/generate")
def generate(body: GenerateIn, db: Session = Depends(get_db), settings=Depends(get_settings)):
    """Ask System 2 to design categories from a processed file's text; merge into (or replace) the current taxonomy."""
    if not settings.system2_enabled:
        raise ValidationFailed("System 2 is switched off.", hint="Set OPENJEV_SYSTEM2_ENABLED=true and make sure Ollama is running.")
    settings.profile(body.profile)
    if db.get(FileRun, body.file_id) is None:
        raise NotFound("That file no longer exists.")
    texts = list(db.scalars(select(RowResult.narration).where(RowResult.file_run_id == body.file_id, RowResult.narration != "").limit(5000)))
    if not texts:
        raise ValidationFailed("That file has no text to learn from.")
    proposed = system2.generate_taxonomy(texts, settings, body.profile)
    if body.mode == "merge":
        cur = tax.load_current(db, settings, body.profile)
        proposed = tax.merge(cur[0] if cur else {}, proposed, settings, body.profile)
    ver, warnings = tax.save(db, settings, body.profile, proposed, source="merge" if body.mode == "merge" else "system2",
                             note=f"{'Merged with' if body.mode == 'merge' else 'Designed by'} {settings.system2_model}")
    return {"version": ver.version, "taxonomy": json.loads(ver.content_json), "warnings": warnings}


@router.post("/api/taxonomy/restore")
def restore(body: RestoreIn, db: Session = Depends(get_db), settings=Depends(get_settings)):
    ver = tax.restore(db, settings, body.profile, body.version_id)
    return {"version": ver.version, "taxonomy": json.loads(ver.content_json), "warnings": []}


@router.get("/taxonomy/download")
def download(profile: str, db: Session = Depends(get_db), settings=Depends(get_settings)):
    settings.profile(profile)
    if tax.load_current(db, settings, profile) is None:
        raise NotFound("This profile has no taxonomy yet.", hint="Create one on the Taxonomy page first.")
    return FileResponse(tax.taxonomy_path(settings, profile), filename=f"{profile}_taxonomy.json", media_type="application/json")
