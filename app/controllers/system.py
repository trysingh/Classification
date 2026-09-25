"""system.py — Diagnostics page and machine-readable health endpoint."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_jobs, get_registry, get_settings
from app.core.templating import render
from app.services import diagnostics_service as diag

router = APIRouter()
_HIDE = {"taxonomy_generation_prompt", "main_category_question", "sub_category_question", "dynamic_label_prompt", "profiles"}


@router.get("/diagnostics", response_class=HTMLResponse)
def page(request: Request, db: Session = Depends(get_db), settings=Depends(get_settings),
         registry=Depends(get_registry), jobs=Depends(get_jobs)):
    return render(request, "diagnostics.html", report=diag.health(settings, registry, jobs, db),
                  errors=diag.recent_errors(db), log_tail=diag.tail_log(settings),
                  config=settings.model_dump(mode="json", exclude=_HIDE), profile_keys=list(settings.profiles))


@router.get("/health")
def health(db: Session = Depends(get_db), settings=Depends(get_settings), registry=Depends(get_registry), jobs=Depends(get_jobs)):
    """For load balancers / monitoring."""
    return diag.health(settings, registry, jobs, db)
