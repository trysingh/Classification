"""jobs.py — job history, live status (polled by the browser) and job controls."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_jobs
from app.core.errors import NotFound
from app.core.templating import render
from app.models.entities import Job
from app.services.analysis_service import job_status

router = APIRouter()


def _job(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise NotFound(f"Job #{job_id} does not exist.", hint="It may have been deleted with its last file.")
    return job


@router.get("/jobs", response_class=HTMLResponse)
def job_list(request: Request, db: Session = Depends(get_db)):
    jobs = db.scalars(select(Job).order_by(Job.id.desc()).limit(50)).all()
    return render(request, "jobs.html", jobs=[job_status(j) | {"created": j.created_at, "ended": j.finished_at}
                                              for j in jobs])


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int, db: Session = Depends(get_db)):
    job = _job(db, job_id)
    return render(request, "job_detail.html", job=job, data=job_status(job))


@router.get("/api/jobs/active")
def active(db: Session = Depends(get_db)):
    """Powers the 'job running' pill shown on every page."""
    jobs = [job_status(j) for j in db.scalars(select(Job).where(Job.status.in_(["queued", "running"])))]
    return {"count": len(jobs), "percent": round(sum(j["percent"] for j in jobs) / len(jobs), 1) if jobs else 0,
            "jobs": [{"id": j["id"], "percent": j["percent"]} for j in jobs]}


@router.get("/api/jobs/{job_id}/status")
def status(job_id: int, db: Session = Depends(get_db)):
    return job_status(_job(db, job_id))


@router.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: int, db: Session = Depends(get_db), jobs=Depends(get_jobs)):
    return job_status(jobs.cancel(db, job_id))


@router.post("/api/jobs/{job_id}/retry")
def retry(job_id: int, db: Session = Depends(get_db), jobs=Depends(get_jobs)):
    return job_status(jobs.retry(db, job_id))
