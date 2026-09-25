"""suppliers.py — maintain the supplier records that files are traced back to."""
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import AppError
from app.core.templating import render
from app.models.entities import Supplier
from app.services import supplier_service

router = APIRouter()


def _back(msg: str, kind: str = "ok", extra: str = "") -> RedirectResponse:
    return RedirectResponse(f"/suppliers?{kind}={quote(msg)}{extra}", status_code=303)   # POST -> redirect -> GET


@router.get("/suppliers", response_class=HTMLResponse)
def page(request: Request, edit: str = "", db: Session = Depends(get_db)):
    editing = db.get(Supplier, int(edit)) if edit.isdigit() else None
    return render(request, "suppliers.html", suppliers=supplier_service.list_with_stats(db),
                  editing=supplier_service.to_dict(editing) if editing else None)


@router.post("/suppliers/save")
def save(id: str = Form(""), name: str = Form(""), code: str = Form(""), country: str = Form(""),
         contact_email: str = Form(""), notes: str = Form(""), db: Session = Depends(get_db)):
    try:
        supplier_service.upsert(db, dict(name=name, code=code, country=country, contact_email=contact_email, notes=notes),
                                int(id) if id.isdigit() else None)
    except AppError as e:                                        # show the problem on the form, not on an error page
        return _back(e.message, "err", f"&edit={id}" if id else "")
    return _back("Supplier saved.")


@router.post("/suppliers/{supplier_id}/delete")
def delete(supplier_id: int, db: Session = Depends(get_db)):
    try:
        supplier_service.delete(db, supplier_id)
    except AppError as e:
        return _back(e.message, "err")
    return _back("Supplier deleted.")
