"""
supplier_service.py
-------------------
Suppliers are captured once and reused: entering 'Acme Ltd' again updates the same record
instead of creating a duplicate, so every file can be traced back to one supplier.
"""
from __future__ import annotations

import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import Conflict, NotFound, ValidationFailed
from app.models.entities import FileRun, Supplier

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
FIELDS = ("code", "country", "contact_email", "notes")


def to_dict(s: Supplier) -> dict:
    return {"id": s.id, "name": s.name, "code": s.code or "", "country": s.country or "",
            "contact_email": s.contact_email or "", "notes": s.notes or ""}


def upsert(db: Session, data: dict, supplier_id: int | None = None) -> Supplier:
    """Create or update by id (edit form) or by case-insensitive name (upload forms)."""
    name = " ".join(str(data.get("name", "")).split())
    if not name:
        raise ValidationFailed("Supplier name is required.")
    if len(name) > 200:
        raise ValidationFailed("Supplier name is too long (max 200 characters).")
    email = str(data.get("contact_email", "")).strip()
    if email and not _EMAIL.match(email):
        raise ValidationFailed(f"'{email}' is not a valid email address.")

    sup = db.get(Supplier, supplier_id) if supplier_id else db.scalars(select(Supplier).where(Supplier.name == name)).first()
    if supplier_id and sup is None:
        raise NotFound("Supplier not found.")
    clash = db.scalars(select(Supplier).where(Supplier.name == name)).first()
    if clash and sup and clash.id != sup.id:
        raise Conflict(f"Another supplier is already called '{name}'.")
    if sup is None:
        sup = Supplier(name=name)
        db.add(sup)
    sup.name = name
    for f in FIELDS:
        v = str(data.get(f, "") or "").strip()
        if v or supplier_id:                                     # edit form may clear a field; upload forms never blank one out
            setattr(sup, f, v[:2000] or None)
    db.commit()
    return sup


def list_with_stats(db: Session) -> list[dict]:
    counts = dict(db.execute(select(FileRun.supplier_id, func.count()).group_by(FileRun.supplier_id)).all())
    rows = db.scalars(select(Supplier).order_by(Supplier.name)).all()
    return [{**to_dict(s), "files": counts.get(s.id, 0)} for s in rows]


def delete(db: Session, supplier_id: int) -> None:
    sup = db.get(Supplier, supplier_id)
    if sup is None:
        raise NotFound("Supplier not found.")
    if db.scalar(select(func.count()).select_from(FileRun).where(FileRun.supplier_id == supplier_id)):
        raise Conflict(f"'{sup.name}' has processed files and cannot be deleted.",
                       hint="Delete or re-assign those files first.")
    db.delete(sup)
    db.commit()
