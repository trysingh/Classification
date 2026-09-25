"""
file_service.py
---------------
Everything about getting data in: safe uploads, reading CSV/Excel into a DataFrame,
guessing which column holds the text, and scanning batch folders. Kept free of DB and
model code so it can be tested and reused on its own.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.errors import FileReadError, NotFound, ValidationFailed

_SAFE = re.compile(r"[^A-Za-z0-9._ -]+")
_NUM = re.compile(r"[^\d.,()\-]")


# ---------------------------------------------------------------- uploads
def safe_filename(name: str) -> str:
    """Strip client paths and odd characters; never trust a user-supplied name."""
    name = _SAFE.sub("_", Path(name or "").name).strip(" .") or "file"
    return name[:120]


def check_extension(name: str, settings) -> None:
    ext = Path(name or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValidationFailed(f"'{ext or name}' is not a supported file type.",
                               hint=f"Supported: {', '.join(settings.allowed_extensions)}")


def save_upload(upload, dest_dir: Path, settings, *, prefix: str | None = None) -> Path:
    """Stream an UploadFile to disk with a size cap. `prefix` -> '<prefix>_<name>'; without it the
    original name is kept (numbered if it already exists), as batch folders expect."""
    check_extension(upload.filename, settings)
    name = safe_filename(upload.filename)
    if prefix:
        path = dest_dir / f"{prefix}_{name}"
    else:
        path, n = dest_dir / name, 1
        while path.exists():
            path, n = dest_dir / f"{Path(name).stem}_{n}{Path(name).suffix}", n + 1
    limit, size = settings.max_upload_mb * 1024 * 1024, 0
    try:
        with open(path, "wb") as fh:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValidationFailed(f"'{name}' is larger than {settings.max_upload_mb} MB.",
                                           hint="Split the file or raise OPENJEV_MAX_UPLOAD_MB.")
                fh.write(chunk)
        if size == 0:
            raise ValidationFailed(f"'{name}' is empty.")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def find_upload(upload_id: str, settings) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id or ""):
        raise ValidationFailed("Invalid upload reference.")
    hits = list(settings.upload_dir.glob(f"{upload_id}_*"))
    if not hits:
        raise NotFound("The uploaded file is no longer available.", hint="Upload it again.")
    return hits[0]


# ---------------------------------------------------------------- reading
def _read_csv(path: Path, sep: str, nrows: int | None) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, nrows=nrows, encoding=enc)
            if df.shape[1] == 1 and sep == "," and re.search(r"[;\t]", str(df.columns[0])):  # ';' or tab exports
                df = pd.read_csv(path, sep=";" if ";" in df.columns[0] else "\t", dtype=str,
                                 keep_default_na=False, nrows=nrows, encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last = e
    raise FileReadError("The file's text encoding is not recognised.", hint="Save it as UTF-8 CSV.", detail=str(last))


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read CSV/TSV/Excel as text (no type guessing, blanks stay blank)."""
    ext = path.suffix.lower()
    try:
        if ext in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(path, dtype=str, nrows=nrows, keep_default_na=False,
                               engine="xlrd" if ext == ".xls" else "openpyxl")
        else:
            df = _read_csv(path, "\t" if ext == ".tsv" else ",", nrows)
    except FileReadError:
        raise
    except ImportError as e:
        raise FileReadError(f"Reading {ext} files needs the '{e.name}' package.", hint=f"pip install {e.name}")
    except Exception as e:
        raise FileReadError(f"Could not read '{path.name}'.",
                            hint="Check it is a valid, unprotected CSV or Excel file.",
                            detail=f"{type(e).__name__}: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise FileReadError(f"'{path.name}' has no data rows.")
    return df


def to_number(v) -> float | None:
    """'1,234.50', '(1,234)', '₹ 1.234,50' -> float; anything else -> None."""
    s = str(v if v is not None else "").strip()
    if not s:
        return None
    neg = (s.startswith("(") and s.endswith(")")) or s.startswith("-")
    s = _NUM.sub("", s).strip("()-")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if all(len(p) == 3 for p in parts[1:]) else s.replace(",", ".")
    try:
        n = float(s)
    except ValueError:
        return None
    return -n if neg else n


def guess_column(df: pd.DataFrame, hints: list[str], *, text: bool = False, numeric: bool = False) -> str | None:
    """Pick a column: exact hint match -> partial match -> (text) longest free text / (numeric) must parse as numbers."""
    norm = {c: c.casefold().replace("_", " ").strip() for c in df.columns}
    ok = (lambda c: _numeric_share(df[c]) >= 0.6) if numeric else (lambda c: True)
    for h in hints:
        for c, n in norm.items():
            if n == h.casefold() and ok(c):
                return c
    for h in hints:
        for c, n in norm.items():
            if h.casefold() in n and ok(c):
                return c
    if text:
        best, best_len = None, 0.0
        for c in df.columns:
            s = df[c].astype(str).head(200)
            if s.str.fullmatch(r"[\d.,\s()\-]*").mean() < 0.8 and s.str.len().mean() > best_len:
                best, best_len = c, float(s.str.len().mean())
        return best if best_len >= 8 else None
    return None


def _numeric_share(series: pd.Series) -> float:
    vals = [v for v in series.head(50) if str(v).strip()]
    return sum(to_number(v) is not None for v in vals) / len(vals) if vals else 0.0


def preview(path: Path, settings) -> dict:
    """What the UI needs after an upload: columns, first rows, and per-profile column guesses."""
    df = read_table(path)
    guesses = {k: {"text": guess_column(df, p.text_column_hints, text=True),
                   "amount": guess_column(df, p.amount_column_hints, numeric=True) if p.amount_column_hints else None}
               for k, p in settings.profiles.items()}
    return {"rows": len(df), "columns": list(df.columns), "guesses": guesses,
            "sample": df.head(5).astype(str).to_dict("records")}


# ---------------------------------------------------------------- batch folders
def resolve_folder(user_path: str | None, settings) -> Path:
    """Relative paths resolve inside the inbox; anything must sit under an allowed root."""
    base = settings.inbox_dir
    p = Path(user_path.strip()) if user_path and user_path.strip() else base
    p = (p if p.is_absolute() else base / p).resolve()
    roots = [base.resolve(), *[r.resolve() for r in settings.allowed_folder_roots]]
    if not any(p == r or r in p.parents for r in roots):
        raise ValidationFailed("That folder is outside the locations batch mode may read.",
                               hint=f"Allowed: {', '.join(map(str, roots))}. Add more via OPENJEV_ALLOWED_FOLDER_ROOTS.")
    if not p.is_dir():
        raise NotFound(f"Folder not found: {p}", hint="Check the path, or create the folder first.")
    return p


def scan_folder(folder: Path, settings, db, profile_key: str) -> list[dict]:
    """List processable files with a detected text column and whether they were already processed."""
    from sqlalchemy import select
    from app.models.entities import FileRun

    hints = settings.profile(profile_key).text_column_hints
    out = []
    for f in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not f.is_file() or f.name.startswith(("~", ".")) or f.suffix.lower() not in settings.allowed_extensions:
            continue
        st = f.stat()
        item = {"name": f.name, "size": st.st_size, "modified": datetime.fromtimestamp(st.st_mtime).strftime("%d %b %Y, %H:%M"),
                "text_column": None, "error": None, "previous": None}
        try:
            item["text_column"] = guess_column(read_table(f, nrows=50), hints, text=True)
        except FileReadError as e:
            item["error"] = e.message
        prev = db.scalars(select(FileRun).where(FileRun.source_path == str(f), FileRun.file_size == st.st_size,
                                                FileRun.status == "completed").order_by(FileRun.id.desc())).first()
        if prev and abs(prev.file_mtime - st.st_mtime) < 2:
            item["previous"] = {"file_id": prev.id, "profile": prev.profile}
        out.append(item)
    return out


def new_upload_id() -> str:
    return uuid.uuid4().hex
