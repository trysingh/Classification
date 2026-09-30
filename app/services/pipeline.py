"""
pipeline.py
-----------
Processes ONE file end-to-end: read -> taxonomy -> load engine -> classify rows -> write outputs.

Reliability rules that make async jobs safe to leave alone:
  * rows are saved every `commit_every_rows`; re-running a file skips rows already saved (resume)
  * the taxonomy is snapshotted on the file, so a resumed run stays consistent even if the taxonomy is edited meanwhile
  * one bad row never kills the file; only N consecutive engine failures do (fail fast on a broken engine)
  * every failure records message + hint + stack trace on the file and in the Diagnostics error log
"""
from __future__ import annotations

import json 
import logging
import time
import traceback
from pathlib import Path
from typing import Callable

from sqlalchemy import func, select

from app.core.errors import AppError, BackendError, FileReadError
from app.engine.hierarchical import HierarchicalClassifier
from app.models.entities import FileRun, RowResult, Supplier, now
from app.services import analysis_service, taxonomy_service
from app.services.diagnostics_service import record_error
from app.services.file_service import guess_column, read_table, to_number

log = logging.getLogger(__name__)


class _Cancelled(Exception):
    pass


def process_file(sf, settings, registry, file_run_id: int, cancelled: Callable[[], bool]) -> None:
    """Never raises: every outcome (completed / failed / cancelled) is written to the FileRun."""
    t0 = time.perf_counter()
    s = sf()
    try:
        _run(s, settings, registry, file_run_id, cancelled, t0)
        _finish(sf, file_run_id, "completed", "Done", t0, add_duration=False)
    except _Cancelled:
        _finish(sf, file_run_id, "cancelled", "Cancelled", t0)
    except Exception as exc:
        if isinstance(exc, AppError):
            log.warning("File run %s failed: %s", file_run_id, exc.message)
        else:
            log.exception("File run %s failed unexpectedly", file_run_id)
        _finish(sf, file_run_id, "failed", "Failed", t0, exc)
    finally:
        s.close()


def _finish(sf, fid: int, status: str, phase: str, t0: float, exc: Exception | None = None,
            add_duration: bool = True) -> None:
    with sf() as s:                                              # fresh session: the working one may be in a failed state
        fr = s.get(FileRun, fid)
        fr.status, fr.phase = status, phase
        fr.finished_at = now()
        if add_duration:
            fr.duration_sec = (fr.duration_sec or 0.0) + (time.perf_counter() - t0)
        if exc is not None:
            if isinstance(exc, AppError):
                fr.error_message, fr.error_hint = exc.message, exc.hint
            else:
                fr.error_message, fr.error_hint = f"{type(exc).__name__}: {exc}", "See the technical details below or Diagnostics."
            fr.error_trace = "".join(traceback.format_exception(exc))
            if isinstance(exc, AppError) and exc.detail:
                fr.error_trace = f"{exc.detail}\n\n{fr.error_trace}"
            record_error(f"file_run:{fid}", fr.error_message or "", fr.error_trace, ref=f"file{fid}")
        s.commit()


def _column(df, given: str | None, hints: list[str], *, text: bool, numeric: bool = False) -> str | None:
    """Explicit choice must exist; no choice -> auto-detect (text is mandatory, amount optional)."""
    if given:
        if given not in df.columns:
            raise FileReadError(f"Column '{given}' was not found in the file.", hint=f"Columns found: {', '.join(df.columns)}")
        return given
    col = guess_column(df, hints, text=text, numeric=numeric)
    if col is None and text:
        raise FileReadError("Could not work out which column holds the text to classify.",
                            hint=f"Columns found: {', '.join(df.columns)}. Use single-file mode to pick the column, "
                                 "or add its name to text_column_hints for this profile in config.")
    return col


def _context(text: str, rec: dict, ctx_cols: list[str], supplier: str | None, use_supplier: bool) -> str:
    """Text plus any extra signals the profile asks for; identical to the text when there are none."""
    parts = [text]
    if use_supplier and supplier:
        parts.append(f"Supplier: {supplier}")
    parts += [f"{c}: {str(rec.get(c, '')).strip()}" for c in ctx_cols if str(rec.get(c, "")).strip()]
    return " | ".join(parts)


def _commit_buffer(s, fr: FileRun, buf: list, counts: dict) -> None:
    """One checkpoint: persist buffered rows and the counters together, so progress is never ahead of the data."""
    s.add_all(buf)
    buf.clear()
    fr.processed_rows, fr.error_rows, fr.review_rows, fr.others_rows = (
        counts["done"], counts["errors"], counts["review"], counts["others"])
    s.commit()


def _run(s, settings, registry, fid: int, cancelled: Callable[[], bool], t0: float) -> None:
    fr = s.get(FileRun, fid)
    profile = settings.profile(fr.profile)
    others = settings.others_label
    fr.status, fr.phase = "running", "Reading file"
    fr.error_message = fr.error_hint = fr.error_trace = None
    fr.started_at, fr.finished_at = fr.started_at or now(), None
    s.commit()
    if cancelled():
        raise _Cancelled()

    df = read_table(Path(fr.source_path))
    text_col = _column(df, fr.narration_column, profile.text_column_hints, text=True)
    if fr.amount_column is None:                                 # None = auto-detect, "" = user chose no amount column
        amount_col = _column(df, None, profile.amount_column_hints, text=False, numeric=True) \
            if profile.amount_column_hints else None
    else:
        amount_col = _column(df, fr.amount_column, [], text=False) if fr.amount_column else None
    fr.narration_column, fr.amount_column, fr.total_rows = text_col, amount_col or "", len(df)
    s.commit()

    if fr.taxonomy_snapshot:                                     # resumed run: keep the taxonomy it started with
        taxonomy = json.loads(fr.taxonomy_snapshot)
    else:
        fr.phase = "Preparing taxonomy"
        s.commit()
        texts = [t for t in df[text_col].astype(str).str.strip() if t]
        taxonomy, version = taxonomy_service.ensure(s, settings, fr.profile, texts)
        fr.taxonomy_snapshot, fr.taxonomy_version = json.dumps(taxonomy, ensure_ascii=False), version
        s.commit()

    fr.phase = "Loading engine"
    s.commit()
    clf = HierarchicalClassifier(registry.get(fr.backend), taxonomy, settings, fr.profile)

    fid_where = RowResult.file_run_id == fid
    done = set(s.scalars(select(RowResult.row_index).where(fid_where)))
    n = lambda *w: s.scalar(select(func.count()).select_from(RowResult).where(fid_where, *w)) or 0
    counts = {"done": len(done), "errors": n(RowResult.error.is_not(None)), "review": n(RowResult.needs_review.is_(True)),
              "others": n(RowResult.main_category == others, RowResult.error.is_(None))}
    fr.phase = "Classifying"
    s.commit()

    supplier = s.get(Supplier, fr.supplier_id).name if fr.supplier_id else None
    ctx_cols = [c for c in profile.context_columns if c in df.columns]
    threshold, cache, buf, consecutive = profile.low_confidence_threshold, {}, [], 0

    try:
        for i, rec in enumerate(df.to_dict("records")):
            if i in done:
                continue
            if cancelled():
                raise _Cancelled()
            text = str(rec.get(text_col, "")).strip()
            row = dict(file_run_id=fid, row_index=i, narration=text,
                       amount=to_number(rec.get(amount_col)) if amount_col else None,
                       source_json=json.dumps(rec, ensure_ascii=False) if settings.store_source_row else None)
            if not text:                                         # data-quality issue, not an engine failure
                row |= dict(main_category="Unclassified", sub_category="", error="Empty text: nothing to classify")
            else:
                ctx = _context(text, rec, ctx_cols, supplier, profile.use_supplier_context)
                try:
                    res = cache.get(ctx) if settings.dedupe_identical_rows else None
                    cached = res is not None
                    if res is None:
                        res = clf.classify(text, ctx)
                        if settings.dedupe_identical_rows:
                            cache[ctx] = res
                    review = res["main_confidence"] < threshold or 0 < res["sub_confidence"] < threshold
                    row |= dict(main_category=res["main_category"], main_confidence=res["main_confidence"],
                                main_top3=res["main_top3"], sub_category=res["sub_category"],
                                sub_confidence=res["sub_confidence"], sub_top3=res["sub_top3"], needs_review=review,
                                latency_sec=0.0 if cached else res["time_taken_sec"])
                    consecutive = 0
                except Exception as e:
                    consecutive += 1
                    log.warning("Row %d of file %d failed: %s", i + 1, fid, e)
                    row |= dict(main_category="Unclassified", sub_category="", error=f"{type(e).__name__}: {e}"[:500])
                    if consecutive >= settings.max_consecutive_row_errors:
                        raise BackendError(f"Stopped after {consecutive} consecutive rows failed to classify.",
                                           hint="The engine looks broken or unreachable. Fix it, then use Retry: "
                                                "rows already saved are kept.", detail=row["error"])
            buf.append(RowResult(**row))
            counts["done"] += 1
            counts["errors"] += bool(row.get("error"))
            counts["review"] += bool(row.get("needs_review"))
            counts["others"] += row.get("main_category") == others and not row.get("error")
            if len(buf) >= settings.commit_every_rows:
                _commit_buffer(s, fr, buf, counts)
        _commit_buffer(s, fr, buf, counts)
    except BaseException:                                        # keep partial progress for cancel / failure / retry
        try:
            _commit_buffer(s, fr, buf, counts)
        except Exception:
            s.rollback()
        raise

    fr.phase = "Writing output"
    fr.finished_at = now()
    fr.duration_sec = (fr.duration_sec or 0.0) + (time.perf_counter() - t0)   # so the JSON header carries it
    s.commit()
    analysis_service.write_outputs(s, settings, fr)
    s.commit()
