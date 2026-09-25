"""single.py — on-demand mode: upload one file, capture supplier details, run it as a background job."""
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_jobs, get_registry, get_settings
from app.core.errors import ValidationFailed
from app.core.templating import render
from app.models.entities import Supplier
from app.services import supplier_service
from app.services.file_service import find_upload, new_upload_id, preview, save_upload
from app.services.job_manager import create_job

router = APIRouter()


class SupplierIn(BaseModel):
    name: str = ""
    code: str = ""
    country: str = ""
    contact_email: str = ""
    notes: str = ""


class SingleSubmit(BaseModel):
    upload_id: str
    profile: str
    backend: str | None = None
    narration_column: str
    amount_column: str = ""              # "" = no amount column
    supplier: SupplierIn = SupplierIn()
    client_token: str | None = None      # makes a double click / retry create only one job


@router.get("/single", response_class=HTMLResponse)
def page(request: Request, db: Session = Depends(get_db), settings=Depends(get_settings), registry=Depends(get_registry)):
    return render(request, "single.html",
                  profiles={k: p.label for k, p in settings.profiles.items()},
                  default_profile=settings.default_profile, backends=registry.availability(),
                  default_backend=settings.classifier_backend,
                  suppliers=[supplier_service.to_dict(s) for s in db.scalars(select(Supplier).order_by(Supplier.name))])


@router.post("/api/single/upload")
def upload(file: UploadFile = File(...), settings=Depends(get_settings)):
    """Save the file immediately and return a preview, so the user maps columns before running."""
    upload_id = new_upload_id()
    path = save_upload(file, settings.upload_dir, settings, prefix=upload_id)
    try:
        info = preview(path, settings)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"upload_id": upload_id, "filename": file.filename, "size": path.stat().st_size, **info}


@router.post("/api/single/submit")
def submit(body: SingleSubmit, db: Session = Depends(get_db), settings=Depends(get_settings), jobs=Depends(get_jobs)):
    settings.profile(body.profile)
    sup = None
    if body.supplier.name.strip():
        sup = supplier_service.upsert(db, body.supplier.model_dump())
    elif settings.require_supplier_single:
        raise ValidationFailed("Supplier name is required.",
                               hint="Enter the supplier this file came from (set require_supplier_single=false in config to make it optional).")
    path = find_upload(body.upload_id, settings)
    job, created = create_job(
        db, settings, kind="single", profile=body.profile, backend=body.backend or None,
        items=[dict(path=path, filename=path.name.split("_", 1)[1], supplier_id=sup.id if sup else None,
                    narration_column=body.narration_column, amount_column=body.amount_column)],
        client_token=body.client_token)
    if created:
        jobs.enqueue(job)
    return {"job_id": job.id, "file_id": job.files[0].id}
