"""
entities.py
-----------
The Model layer: every piece of structured information the product persists.

Supplier -> FileRun (one processed file) -> RowResult (one classified row)
Job groups FileRuns (kind 'single' = on-demand, 'batch' = folder run).
TaxonomyVersion keeps the audit trail of every taxonomy change; ErrorLog feeds Diagnostics.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now() -> datetime:
    return datetime.now()


class Base(DeclarativeBase):
    pass


class Supplier(Base):
    """Who supplied a file. Name is unique (case-insensitive) so suppliers are reused, not duplicated."""
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200, collation="NOCASE"), unique=True)
    code: Mapped[Optional[str]] = mapped_column(String(80))
    country: Mapped[Optional[str]] = mapped_column(String(80))
    contact_email: Mapped[Optional[str]] = mapped_column(String(200))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)
    files: Mapped[list["FileRun"]] = relationship(back_populates="supplier")


class Job(Base):
    """A unit of async work the user can leave and come back to. Survives restarts (state lives here)."""
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(10))                    # single | batch
    status: Mapped[str] = mapped_column(String(24), default="queued")  # queued|running|completed|completed_with_errors|failed|cancelled
    profile: Mapped[str] = mapped_column(String(60))
    backend: Mapped[str] = mapped_column(String(20))
    folder: Mapped[Optional[str]] = mapped_column(String(500))
    client_token: Mapped[Optional[str]] = mapped_column(String(64), unique=True)  # makes re-submits idempotent
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    files: Mapped[list["FileRun"]] = relationship(back_populates="job", cascade="all, delete-orphan",
                                                  order_by="FileRun.id")


class FileRun(Base):
    """One file processed once: its inputs, progress counters, taxonomy used and output locations."""
    __tablename__ = "file_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    supplier_id: Mapped[Optional[int]] = mapped_column(ForeignKey("suppliers.id", ondelete="SET NULL"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    source_path: Mapped[str] = mapped_column(String(600))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    file_mtime: Mapped[float] = mapped_column(Float, default=0.0)
    profile: Mapped[str] = mapped_column(String(60))
    backend: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|completed|failed|cancelled
    phase: Mapped[str] = mapped_column(String(60), default="Queued")
    narration_column: Mapped[Optional[str]] = mapped_column(String(200))
    amount_column: Mapped[Optional[str]] = mapped_column(String(200))   # None = auto-detect, "" = none
    total_rows: Mapped[Optional[int]] = mapped_column(Integer)
    processed_rows: Mapped[int] = mapped_column(Integer, default=0)
    error_rows: Mapped[int] = mapped_column(Integer, default=0)
    review_rows: Mapped[int] = mapped_column(Integer, default=0)
    others_rows: Mapped[int] = mapped_column(Integer, default=0)
    taxonomy_version: Mapped[Optional[int]] = mapped_column(Integer)
    taxonomy_snapshot: Mapped[Optional[str]] = mapped_column(Text)      # taxonomy as used -> reproducible results
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_hint: Mapped[Optional[str]] = mapped_column(Text)
    error_trace: Mapped[Optional[str]] = mapped_column(Text)
    output_json_path: Mapped[Optional[str]] = mapped_column(String(600))
    output_csv_path: Mapped[Optional[str]] = mapped_column(String(600))
    job: Mapped["Job"] = relationship(back_populates="files")
    supplier: Mapped[Optional["Supplier"]] = relationship(back_populates="files")


class RowResult(Base):
    """One classified row. Written in checkpoints so a crashed run resumes instead of restarting."""
    __tablename__ = "row_results"
    __table_args__ = (UniqueConstraint("file_run_id", "row_index"), Index("ix_rows_file_main", "file_run_id", "main_category"))
    id: Mapped[int] = mapped_column(primary_key=True)
    file_run_id: Mapped[int] = mapped_column(ForeignKey("file_runs.id", ondelete="CASCADE"), index=True)
    row_index: Mapped[int] = mapped_column(Integer)
    narration: Mapped[str] = mapped_column(Text, default="")
    amount: Mapped[Optional[float]] = mapped_column(Float)
    main_category: Mapped[str] = mapped_column(String(200), default="")
    main_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    main_top3: Mapped[Optional[str]] = mapped_column(Text)
    sub_category: Mapped[str] = mapped_column(String(200), default="")
    sub_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sub_top3: Mapped[Optional[str]] = mapped_column(Text)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_sec: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text)
    source_json: Mapped[Optional[str]] = mapped_column(Text)            # original row, all columns


class TaxonomyVersion(Base):
    """Every saved taxonomy, per profile. The live file on disk is the source of truth; this is history."""
    __tablename__ = "taxonomy_versions"
    __table_args__ = (UniqueConstraint("profile", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    profile: Mapped[str] = mapped_column(String(60), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_json: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20))                     # manual|system2|seed|file|file-edit|restore|merge
    note: Mapped[Optional[str]] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ErrorLog(Base):
    """Errors worth showing in Diagnostics (with stack trace), independent of the log file."""
    __tablename__ = "error_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    scope: Mapped[str] = mapped_column(String(80))
    message: Mapped[str] = mapped_column(Text)
    detail: Mapped[Optional[str]] = mapped_column(Text)
    ref: Mapped[Optional[str]] = mapped_column(String(40))
