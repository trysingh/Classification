"""
analysis_service.py
-------------------
Turns stored row results into what users see and download: per-file summary (category
breakdown, confidence spread, spend), filtered/paged rows, the streamed JSON + CSV outputs,
and the small status payloads the browser polls for progress.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models.entities import FileRun, Job, RowResult
from app.services import supplier_service
from app.services.file_service import safe_filename

log = logging.getLogger(__name__)
FINISHED_JOB = ("completed", "completed_with_errors", "failed", "cancelled")
MAX_SUBS_SHOWN = 12
RESULT_COLS = ["main_category", "main_confidence", "sub_category", "sub_confidence", "needs_review",
               "main_top3", "sub_top3", "latency_sec", "error"]


def _iso(v: datetime | None) -> str | None:
    return v.isoformat(timespec="seconds") if v else None


# ---------------------------------------------------------------- live status (polled by the browser)
def file_progress(fr: FileRun) -> float:
    if fr.status == "completed":
        return 100.0
    return round(100.0 * fr.processed_rows / fr.total_rows, 1) if fr.total_rows else 0.0


def file_status(fr: FileRun) -> dict:
    return {"id": fr.id, "filename": fr.filename, "status": fr.status, "phase": fr.phase,
            "supplier": fr.supplier.name if fr.supplier else None,
            "processed_rows": fr.processed_rows, "total_rows": fr.total_rows, "percent": file_progress(fr),
            "review_rows": fr.review_rows, "others_rows": fr.others_rows, "error_rows": fr.error_rows,
            "error_message": fr.error_message, "error_hint": fr.error_hint, "error_trace": fr.error_trace}


def job_status(job: Job) -> dict:
    files = [file_status(f) for f in job.files]
    by = lambda st: sum(f["status"] == st for f in files)
    return {"id": job.id, "kind": job.kind, "status": job.status, "finished": job.status in FINISHED_JOB,
            "profile": job.profile, "backend": job.backend, "cancel_requested": job.cancel_requested,
            "created_at": _iso(job.created_at), "error_message": job.error_message, "files": files,
            "percent": round(sum(f["percent"] for f in files) / len(files), 1) if files else 0.0,
            "counts": {"files": len(files), "completed": by("completed"), "failed": by("failed"),
                       "running": by("running"), "queued": by("queued"), "cancelled": by("cancelled")}}


# ---------------------------------------------------------------- per-file analysis
def summary(db: Session, fr: FileRun, settings) -> dict:
    base = RowResult.file_run_id == fr.id
    ok = (base, RowResult.error.is_(None))
    others = settings.others_label
    count = lambda *w: db.scalar(select(func.count()).select_from(RowResult).where(*w)) or 0

    groups = db.execute(
        select(RowResult.main_category, RowResult.sub_category, func.count(),
               func.coalesce(func.sum(RowResult.amount), 0.0), func.coalesce(func.sum(RowResult.main_confidence), 0.0),
               func.coalesce(func.sum(case((RowResult.needs_review.is_(True), 1), else_=0)), 0))
        .where(*ok).group_by(RowResult.main_category, RowResult.sub_category)).all()

    mains: dict[str, dict] = {}
    for main, sub, n, amt, conf_sum, rev in groups:
        m = mains.setdefault(main, {"main": main, "count": 0, "amount": 0.0, "conf_sum": 0.0, "review": 0, "subs": []})
        m["count"] += n; m["amount"] += amt; m["conf_sum"] += conf_sum; m["review"] += rev
        m["subs"].append({"sub": sub or "(none)", "count": n, "amount": amt})
    classified = sum(m["count"] for m in mains.values())
    by_main = sorted(mains.values(), key=lambda m: (m["main"] == others, -m["count"]))
    for m in by_main:
        m["pct"] = m["count"] / classified if classified else 0.0
        m["avg_conf"] = m["conf_sum"] / m["count"] if m["count"] else 0.0
        m["subs"].sort(key=lambda s: -s["count"])
        m["subs_more"] = max(0, len(m["subs"]) - MAX_SUBS_SHOWN)
        m["subs"] = m["subs"][:MAX_SUBS_SHOWN]

    bucket = case((RowResult.main_confidence < 0.5, "low"), (RowResult.main_confidence < 0.7, "fair"),
                  (RowResult.main_confidence < 0.9, "good"), else_="high")
    buckets = dict(db.execute(select(bucket, func.count()).where(*ok).group_by(bucket)).all())
    return {
        "saved": count(base), "classified": classified, "errors": count(base, RowResult.error.is_not(None)),
        "review": count(base, RowResult.needs_review.is_(True)),
        "others": mains[others]["count"] if others in mains else 0,
        "spend": db.scalar(select(func.sum(RowResult.amount)).where(*ok)) if fr.amount_column else None,
        "avg_conf": (sum(m["conf_sum"] for m in by_main) / classified) if classified else 0.0,
        "by_main": by_main, "buckets": {k: buckets.get(k, 0) for k in ("low", "fair", "good", "high")},
    }


def rows_page(db: Session, file_id: int, *, q: str = "", main: str = "", review: bool = False,
              errors: bool = False, page: int = 1, page_size: int = 25) -> dict:
    stmt = select(RowResult).where(RowResult.file_run_id == file_id)
    if q:
        stmt = stmt.where(RowResult.narration.ilike(f"%{q}%"))
    if main:
        stmt = stmt.where(RowResult.main_category == main)
    if review:
        stmt = stmt.where(RowResult.needs_review.is_(True))
    if errors:
        stmt = stmt.where(RowResult.error.is_not(None))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    pages = max(1, -(-total // page_size))
    page = min(max(1, page), pages)
    items = db.scalars(stmt.order_by(RowResult.row_index).limit(page_size).offset((page - 1) * page_size)).all()
    return {"items": items, "total": total, "page": page, "pages": pages}


# ---------------------------------------------------------------- exports
def _row_dict(r: RowResult) -> dict:
    return {"row": r.row_index + 1, "text": r.narration, "amount": r.amount,
            "main_category": r.main_category, "main_confidence": r.main_confidence,
            "sub_category": r.sub_category, "sub_confidence": r.sub_confidence, "needs_review": r.needs_review,
            "main_top3": r.main_top3, "sub_top3": r.sub_top3, "latency_sec": r.latency_sec, "error": r.error,
            "source": json.loads(r.source_json) if r.source_json else None}


def build_header(db: Session, settings, fr: FileRun) -> dict:
    """Everything in the JSON output except the row list (which is streamed)."""
    s = summary(db, fr, settings)
    model = {"causal_lm": settings.system1_model, "laya": settings.laya_model_name}.get(fr.backend, fr.backend)
    return {
        "file": {"id": fr.id, "name": fr.filename, "profile": fr.profile, "engine": fr.backend, "model": model,
                 "text_column": fr.narration_column, "amount_column": fr.amount_column or None,
                 "rows": fr.total_rows, "taxonomy_version": fr.taxonomy_version, "started_at": _iso(fr.started_at),
                 "finished_at": _iso(fr.finished_at), "duration_sec": round(fr.duration_sec or 0, 2)},
        "supplier": supplier_service.to_dict(fr.supplier) if fr.supplier else None,
        "summary": {
            "rows_classified": s["classified"], "rows_with_errors": s["errors"], "rows_needing_review": s["review"],
            "rows_in_others": s["others"], "total_amount": s["spend"], "average_confidence": round(s["avg_conf"], 4),
            "by_category": [{"category": m["main"], "rows": m["count"], "share": round(m["pct"], 4),
                             "amount": round(m["amount"], 2), "average_confidence": round(m["avg_conf"], 4),
                             "sub_categories": [{"name": x["sub"], "rows": x["count"], "amount": round(x["amount"], 2)}
                                                for x in m["subs"]]} for m in s["by_main"]]},
        "taxonomy": json.loads(fr.taxonomy_snapshot) if fr.taxonomy_snapshot else None,
        "generated_at": _iso(datetime.now()),
    }


def write_outputs(db: Session, settings, fr: FileRun) -> tuple[Path, Path]:
    """Stream results to <id>_<name>.json and .csv without holding the whole file in memory."""
    stem = f"{fr.id:05d}_{safe_filename(Path(fr.filename).stem)[:60]}"
    jpath, cpath = settings.output_dir / f"{stem}.json", settings.output_dir / f"{stem}.csv"
    header = build_header(db, settings, fr)
    rows = lambda: db.execute(select(RowResult).where(RowResult.file_run_id == fr.id).order_by(RowResult.row_index)
                              .execution_options(yield_per=1000)).scalars()

    with open(jpath, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        for k, v in header.items():
            fh.write(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, default=str)},\n")
        fh.write('  "results": [\n')
        first = True
        for r in rows():
            fh.write(("" if first else ",\n") + "    " + json.dumps(_row_dict(r), ensure_ascii=False, default=str))
            first = False
        fh.write("\n  ]\n}\n")

    sample = db.scalars(select(RowResult.source_json).where(RowResult.file_run_id == fr.id,
                                                            RowResult.source_json.is_not(None)).limit(1)).first()
    src_cols = list(json.loads(sample)) if sample else []
    with open(cpath, "w", newline="", encoding="utf-8-sig") as fh:      # BOM so Excel opens UTF-8 correctly
        w = csv.writer(fh)
        w.writerow(["row", *(src_cols or ["text", "amount"]), *RESULT_COLS])
        for r in rows():
            src = json.loads(r.source_json) if r.source_json else {}
            lead = [src.get(c, "") for c in src_cols] if src_cols else [r.narration, r.amount]
            w.writerow([r.row_index + 1, *lead, *[getattr(r, c) for c in RESULT_COLS]])

    fr.output_json_path, fr.output_csv_path = str(jpath), str(cpath)
    return jpath, cpath


def delete_outputs(fr: FileRun) -> None:
    for p in (fr.output_json_path, fr.output_csv_path):
        if p:
            Path(p).unlink(missing_ok=True)
