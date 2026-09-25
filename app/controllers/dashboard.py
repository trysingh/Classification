"""dashboard.py — landing page: what is running, what finished, where to start."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.templating import render
from app.models.entities import FileRun, Job, Supplier
from app.services.analysis_service import job_status

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, db: Session = Depends(get_db)):
    done = FileRun.status == "completed"
    count = lambda col, *w: db.scalar(select(func.coalesce(func.sum(col), 0)).where(*w)) or 0
    stats = {
        "files": db.scalar(select(func.count()).select_from(FileRun).where(done)) or 0,
        "rows": count(FileRun.processed_rows, done),
        "review": count(FileRun.review_rows, done),
        "suppliers": db.scalar(select(func.count()).select_from(Supplier)) or 0,
        "failed": db.scalar(select(func.count()).select_from(FileRun).where(FileRun.status == "failed")) or 0,
    }
    recent = db.scalars(select(FileRun).order_by(FileRun.id.desc()).limit(8)).all()
    active = db.scalars(select(Job).where(Job.status.in_(["queued", "running"])).order_by(Job.id.desc())).all()
    return render(request, "dashboard.html", stats=stats, recent=recent, active=[job_status(j) for j in active])
