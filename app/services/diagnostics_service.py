"""
diagnostics_service.py
----------------------
Everything needed to debug a live system from the browser: persisted errors with stack
traces, health checks (DB, folders, Ollama, engines, workers) and the tail of the log file.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.database import get_sf
from app.core.logging_setup import request_id_var
from app.engine import system2
from app.models.entities import ErrorLog, FileRun, Job

log = logging.getLogger(__name__)


def record_error(scope: str, message: str, detail: str | None = None, ref: str | None = None) -> None:
    """Persist an error for the Diagnostics page. Must never raise: it runs inside failure paths."""
    try:
        with get_sf()() as s:
            s.add(ErrorLog(scope=scope[:80], message=message[:2000], detail=(detail or "")[:20000],
                           ref=(ref or request_id_var.get())[:40]))
            s.commit()
    except Exception:
        log.exception("Could not write to error_log")


def recent_errors(db: Session, limit: int = 25) -> list[ErrorLog]:
    return list(db.scalars(select(ErrorLog).order_by(ErrorLog.id.desc()).limit(limit)))


def tail_log(settings, lines: int = 120) -> str:
    path = settings.log_dir / "app.log"
    if not path.exists():
        return "(no log file yet)"
    with open(path, "rb") as fh:                                 # read only the end of the file
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 64_000))
        return "\n".join(fh.read().decode("utf-8", "replace").splitlines()[-lines:])


def health(settings, registry, jobs, db: Session) -> dict:
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    try:
        db.execute(text("SELECT 1"))
        add("Database", "ok", str(settings.db_path))
    except Exception as e:
        add("Database", "fail", f"{type(e).__name__}: {e}")

    for label, path in (("Batch inbox folder", settings.inbox_dir), ("Uploads folder", settings.upload_dir),
                        ("Outputs folder", settings.output_dir), ("Taxonomy folder", settings.taxonomy_dir)):
        probe = path / ".write_test"
        try:
            probe.write_text("ok")
            probe.unlink()
            add(label, "ok", str(path))
        except Exception as e:
            add(label, "fail", f"{path} is not writable: {e}")

    alive = jobs.alive_workers()
    add("Background workers", "ok" if alive else "fail", f"{alive} thread(s) running")

    if settings.system2_enabled:
        st = system2.ollama_status(settings)
        if not st["reachable"]:
            add("System 2 (Ollama)", "warn", f"Not reachable at {settings.ollama_base_url}. Seed taxonomies still work.")
        elif not st["model_available"]:
            add("System 2 (Ollama)", "warn", f"Reachable, but model '{settings.system2_model}' is not pulled.")
        else:
            add("System 2 (Ollama)", "ok", f"{settings.system2_model} at {settings.ollama_base_url}")
    else:
        add("System 2 (Ollama)", "warn", "Disabled (OPENJEV_SYSTEM2_ENABLED=false)")

    for b in registry.availability():
        add(f"Engine: {b['label']}", "ok" if b["installed"] else "warn",
            ("loaded" if b["loaded"] else "installed, loads on first use") if b["installed"] else f"needs {b['needs']}")

    counts = {
        "files": db.scalar(select(func.count()).select_from(FileRun)) or 0,
        "running_jobs": db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(["queued", "running"]))) or 0,
    }
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"status": worst, "checked_at": datetime.now().isoformat(timespec="seconds"), "checks": checks, "counts": counts}
