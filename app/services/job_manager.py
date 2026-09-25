"""
job_manager.py
--------------
Async processing that survives closed tabs, network drops and server restarts.

The DB is the source of truth (Job / FileRun rows). Worker threads only pull job ids from
in-memory queues; on startup every unfinished job is re-queued and each file resumes from its
last checkpoint. On-demand ('single') and folder ('batch') jobs have separate workers, so a
long batch never blocks a user waiting on one file. The browser just polls the status API.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.errors import Conflict, NotFound, ValidationFailed
from app.models.entities import FileRun, Job, now
from app.services.analysis_service import FINISHED_JOB
from app.services.diagnostics_service import record_error
from app.services.pipeline import process_file

log = logging.getLogger(__name__)


def create_job(db, settings, *, kind: str, profile: str, backend: str | None, items: list[dict],
               folder: str | None = None, client_token: str | None = None) -> tuple[Job, bool]:
    """Persist a job and its files. Returns (job, created); `created` is False when the same
    client_token was already submitted (double click / retry after a dropped connection)."""
    if not items:
        raise ValidationFailed("Select at least one file.")
    if client_token:
        existing = db.scalars(select(Job).where(Job.client_token == client_token)).first()
        if existing:
            return existing, False
    backend = settings.backend_for(profile, backend)            # also validates the profile
    job = Job(kind=kind, profile=profile, backend=backend, folder=folder, client_token=client_token)
    for it in items:
        p = Path(it["path"])
        st = p.stat()
        job.files.append(FileRun(filename=it["filename"], source_path=str(p), file_size=st.st_size,
                                 file_mtime=st.st_mtime, profile=profile, backend=backend,
                                 supplier_id=it.get("supplier_id"), narration_column=it.get("narration_column"),
                                 amount_column=it.get("amount_column")))
    db.add(job)
    try:
        db.commit()
    except IntegrityError:                                       # two identical submits raced: keep the first
        db.rollback()
        return db.scalars(select(Job).where(Job.client_token == client_token)).one(), False
    return job, True


class JobManager:
    def __init__(self, settings, session_factory, registry):
        self.settings, self.sf, self.registry = settings, session_factory, registry
        self._queues = {"single": queue.Queue(), "batch": queue.Queue()}
        self._threads: list[threading.Thread] = []
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ---- lifecycle
    def start(self) -> None:
        self._stop.clear()
        for kind, n in (("single", self.settings.single_workers), ("batch", self.settings.batch_workers)):
            for k in range(max(1, n)):
                t = threading.Thread(target=self._loop, args=(kind,), name=f"worker-{kind}-{k}", daemon=True)
                t.start()
                self._threads.append(t)
        self._recover()

    def stop(self) -> None:
        self._stop.set()

    def alive_workers(self) -> int:
        return sum(t.is_alive() for t in self._threads)

    def _recover(self) -> None:
        """After a restart: unfinished jobs go back on the queue and resume from their checkpoints."""
        with self.sf() as s:
            jobs = list(s.scalars(select(Job).where(Job.status.in_(["queued", "running"])).order_by(Job.id)))
            for j in jobs:
                for f in j.files:
                    if f.status == "running":
                        f.status, f.phase = "queued", "Queued (resuming after restart)"
                j.status = "queued"
            s.commit()
            pending = [(j.id, j.kind) for j in jobs]
        for jid, kind in pending:
            self._queues[kind].put(jid)
        if pending:
            log.info("Recovered %d unfinished job(s) after restart", len(pending))

    # ---- commands (called by controllers)
    def enqueue(self, job: Job) -> None:
        self._queues[job.kind].put(job.id)

    def cancel(self, db, job_id: int) -> Job:
        job = db.get(Job, job_id)
        if job is None:
            raise NotFound("Job not found.")
        if job.status in FINISHED_JOB:
            raise Conflict("This job has already finished.")
        job.cancel_requested = True
        self._event(job_id).set()
        if job.status == "queued":                               # not started yet: close it right away
            for f in job.files:
                if f.status == "queued":
                    f.status, f.phase = "cancelled", "Cancelled"
            job.status, job.finished_at = "cancelled", now()
        db.commit()
        return job

    def retry(self, db, job_id: int) -> Job:
        """Re-run failed/cancelled files. Rows already saved are kept, so work resumes, not restarts."""
        job = db.get(Job, job_id)
        if job is None:
            raise NotFound("Job not found.")
        if job.status not in FINISHED_JOB:
            raise Conflict("This job is still running.")
        todo = [f for f in job.files if f.status in ("failed", "cancelled")]
        if not todo:
            raise Conflict("There are no failed or cancelled files to retry.")
        for f in todo:
            f.status, f.phase = "queued", "Queued"
            f.error_message = f.error_hint = f.error_trace = None
        job.status, job.cancel_requested, job.finished_at, job.error_message = "queued", False, None, None
        with self._lock:
            self._events.pop(job_id, None)
        db.commit()
        self.enqueue(job)
        return job

    # ---- worker internals
    def _event(self, job_id: int) -> threading.Event:
        with self._lock:
            return self._events.setdefault(job_id, threading.Event())

    def _loop(self, kind: str) -> None:
        q = self._queues[kind]
        while not self._stop.is_set():
            try:
                job_id = q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            except Exception as e:                               # process_file never raises; this guards the bookkeeping
                log.exception("Job %s crashed", job_id)
                record_error(f"job:{job_id}", f"{type(e).__name__}: {e}", traceback.format_exc(), ref=f"job{job_id}")
                self._mark_crashed(job_id, e)
            finally:
                q.task_done()

    def _run_job(self, job_id: int) -> None:
        ev = self._event(job_id)
        with self.sf() as s:
            job = s.get(Job, job_id)
            if job is None or job.status in FINISHED_JOB:
                return
            if job.cancel_requested:
                ev.set()
            job.status, job.started_at = "running", job.started_at or now()
            s.commit()
            todo = [f.id for f in job.files if f.status in ("queued", "running")]
        log.info("Job %d started (%d file(s))", job_id, len(todo))
        for fid in todo:
            if ev.is_set():
                break
            process_file(self.sf, self.settings, self.registry, fid, ev.is_set)
        self._finalize(job_id, ev.is_set())

    def _finalize(self, job_id: int, cancelled: bool) -> None:
        with self.sf() as s:
            job = s.get(Job, job_id)
            for f in job.files:
                if cancelled and f.status in ("queued", "running"):
                    f.status, f.phase = "cancelled", "Cancelled"
            c = Counter(f.status for f in job.files)
            if c["completed"] == len(job.files):
                job.status = "completed"
            elif cancelled and c["cancelled"]:
                job.status = "cancelled"
            elif c["completed"] == 0 and c["failed"]:
                job.status = "failed"
            else:
                job.status = "completed_with_errors"
            job.finished_at = now()
            s.commit()
            log.info("Job %d finished: %s", job_id, job.status)
        with self._lock:
            self._events.pop(job_id, None)

    def _mark_crashed(self, job_id: int, exc: Exception) -> None:
        try:
            with self.sf() as s:
                job = s.get(Job, job_id)
                job.status, job.finished_at = "failed", now()
                job.error_message = f"{type(exc).__name__}: {exc}"
                s.commit()
        except Exception:
            log.exception("Could not mark job %s as failed", job_id)
