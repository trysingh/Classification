"""batch.py — folder mode: scan a folder, pick files (+ supplier per file), start one async job."""
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_jobs, get_registry, get_settings
from app.core.errors import NotFound
from app.core.templating import render
from app.models.entities import Supplier
from app.services import supplier_service
from app.services.file_service import check_extension, resolve_folder, save_upload, scan_folder
from app.services.job_manager import create_job

router = APIRouter()


class ScanIn(BaseModel):
    folder: str = ""
    profile: str = ""


class BatchItem(BaseModel):
    name: str
    supplier: str = ""


class BatchSubmit(BaseModel):
    folder: str = ""
    profile: str
    backend: str | None = None
    items: list[BatchItem]
    default_supplier: str = ""           # used for files whose own supplier box is empty
    narration_column: str = ""           # optional override; blank = auto-detect per file
    client_token: str | None = None


@router.get("/batch", response_class=HTMLResponse)
def page(request: Request, db: Session = Depends(get_db), settings=Depends(get_settings), registry=Depends(get_registry)):
    return render(request, "batch.html", folder=str(settings.inbox_dir),
                  profiles={k: p.label for k, p in settings.profiles.items()}, default_profile=settings.default_profile,
                  backends=registry.availability(), default_backend=settings.classifier_backend,
                  suppliers=[s.name for s in db.scalars(select(Supplier).order_by(Supplier.name))])


@router.post("/api/batch/scan")
def scan(body: ScanIn, db: Session = Depends(get_db), settings=Depends(get_settings)):
    folder = resolve_folder(body.folder, settings)
    profile = body.profile or settings.default_profile
    return {"folder": str(folder), "files": scan_folder(folder, settings, db, profile)}


@router.post("/api/batch/upload")
def upload(files: list[UploadFile] = File(...), folder: str = Form(""), profile: str = Form(""),
           db: Session = Depends(get_db), settings=Depends(get_settings)):
    """Convenience for users without access to the server folder: drop files straight into it."""
    dest = resolve_folder(folder, settings)
    saved = [save_upload(f, dest, settings).name for f in files]
    return {"saved": saved, "folder": str(dest),
            "files": scan_folder(dest, settings, db, profile or settings.default_profile)}


@router.post("/api/batch/submit")
def submit(body: BatchSubmit, db: Session = Depends(get_db), settings=Depends(get_settings), jobs=Depends(get_jobs)):
    folder = resolve_folder(body.folder, settings)
    items = []
    for it in body.items:
        name = Path(it.name).name                                # never trust client paths
        path = folder / name
        if not path.is_file():
            raise NotFound(f"'{name}' is no longer in the folder.", hint="Scan the folder again.")
        check_extension(name, settings)
        supplier_name = (it.supplier or body.default_supplier).strip()
        sup = supplier_service.upsert(db, {"name": supplier_name}) if supplier_name else None
        items.append(dict(path=path, filename=name, supplier_id=sup.id if sup else None,
                          narration_column=body.narration_column.strip() or None, amount_column=None))
    job, created = create_job(db, settings, kind="batch", profile=body.profile, backend=body.backend or None,
                              items=items, folder=str(folder), client_token=body.client_token)
    if created:
        jobs.enqueue(job)
    return {"job_id": job.id}
