"""files.py — per-file results: history, analysis view, JSON/CSV downloads, reprocess, delete."""
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_jobs, get_settings
from app.core.errors import Conflict, NotFound
from app.core.templating import render
from app.models.entities import FileRun, Job, Supplier
from app.services import analysis_service
from app.services.job_manager import create_job

router = APIRouter()


def _file(db: Session, file_id: int) -> FileRun:
    fr = db.get(FileRun, file_id)
    if fr is None:
        raise NotFound(f"File result #{file_id} does not exist.", hint="It may have been deleted.")
    return fr


@router.get("/files", response_class=HTMLResponse)
def file_list(request: Request, status: str = "", supplier_id: str = "", profile: str = "",
              db: Session = Depends(get_db), settings=Depends(get_settings)):
    stmt = select(FileRun).order_by(FileRun.id.desc()).limit(200)
    if status:
        stmt = stmt.where(FileRun.status == status)
    if supplier_id.isdigit():
        stmt = stmt.where(FileRun.supplier_id == int(supplier_id))
    if profile:
        stmt = stmt.where(FileRun.profile == profile)
    return render(request, "files.html", files=db.scalars(stmt).all(), status=status, supplier_id=supplier_id,
                  profile=profile, suppliers=db.scalars(select(Supplier).order_by(Supplier.name)).all(),
                  profiles={k: p.label for k, p in settings.profiles.items()})


@router.get("/files/{file_id}", response_class=HTMLResponse)
def file_detail(request: Request, file_id: int, q: str = "", main: str = "", review: str = "", errors: str = "",
                page: int = 1, db: Session = Depends(get_db), settings=Depends(get_settings)):
    fr = _file(db, file_id)
    profile = settings.profiles.get(fr.profile)
    ctx = dict(fr=fr, profile=profile, filters=dict(q=q, main=main, review=review, errors=errors))
    if fr.processed_rows:                                        # partial results are viewable while a run is in progress
        ctx["summary"] = analysis_service.summary(db, fr, settings)
        ctx["rows"] = analysis_service.rows_page(db, file_id, q=q, main=main, review=bool(review), errors=bool(errors),
                                                 page=page, page_size=settings.page_size)
        ctx["qs"] = urlencode({k: v for k, v in dict(q=q, main=main, review=review, errors=errors).items() if v})
    return render(request, "file_detail.html", **ctx)


def _download(fr: FileRun, db: Session, settings, kind: str) -> FileResponse:
    if fr.status != "completed":
        raise Conflict("Results can be downloaded once the file has finished processing.")
    attr = "output_json_path" if kind == "json" else "output_csv_path"
    if not (getattr(fr, attr) and Path(getattr(fr, attr)).exists()):      # e.g. output folder was cleaned: rebuild from the DB
        analysis_service.write_outputs(db, settings, fr)
        db.commit()
    path = Path(getattr(fr, attr))
    return FileResponse(path, filename=path.name, media_type="application/json" if kind == "json" else "text/csv")


@router.get("/files/{file_id}/download.json")
def download_json(file_id: int, db: Session = Depends(get_db), settings=Depends(get_settings)):
    return _download(_file(db, file_id), db, settings, "json")


@router.get("/files/{file_id}/download.csv")
def download_csv(file_id: int, db: Session = Depends(get_db), settings=Depends(get_settings)):
    return _download(_file(db, file_id), db, settings, "csv")


@router.post("/api/files/{file_id}/reprocess")
def reprocess(file_id: int, db: Session = Depends(get_db), settings=Depends(get_settings), jobs=Depends(get_jobs)):
    """Run the same file again from scratch, e.g. after editing the taxonomy. The old result stays for comparison."""
    fr = _file(db, file_id)
    if not Path(fr.source_path).exists():
        raise NotFound("The original file is no longer on disk.", hint="Upload it again.")
    job, _ = create_job(db, settings, kind=fr.job.kind, profile=fr.profile, backend=fr.backend, folder=fr.job.folder,
                        items=[dict(path=fr.source_path, filename=fr.filename, supplier_id=fr.supplier_id,
                                    narration_column=fr.narration_column, amount_column=fr.amount_column)])
    jobs.enqueue(job)
    return {"job_id": job.id, "file_id": job.files[0].id}


@router.post("/api/files/{file_id}/delete")
def delete(file_id: int, db: Session = Depends(get_db), settings=Depends(get_settings)):
    fr = _file(db, file_id)
    if fr.status in ("queued", "running"):
        raise Conflict("This file is still being processed.", hint="Cancel its job first.")
    analysis_service.delete_outputs(fr)
    job_id, src = fr.job_id, Path(fr.source_path)
    db.delete(fr)                                                # row results go with it (ON DELETE CASCADE)
    db.commit()
    if src.parent == settings.upload_dir and not db.scalar(select(func.count()).select_from(FileRun).where(FileRun.source_path == str(src))):
        src.unlink(missing_ok=True)                              # uploaded copy no longer needed
    if not db.scalar(select(func.count()).select_from(FileRun).where(FileRun.job_id == job_id)):
        job = db.get(Job, job_id)
        if job:
            db.delete(job)
            db.commit()
    return {"deleted": file_id}
