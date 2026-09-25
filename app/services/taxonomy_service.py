"""
taxonomy_service.py
-------------------
Owns the taxonomy as DATA. One JSON file per profile (data/taxonomies/<profile>.json) is the
source of truth the engine reads; every change (UI edit, System-2 design, hand-edit of the file,
restore) is validated and recorded as a numbered version, so results can always say which
taxonomy produced them and users can roll back.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import System2Error, TaxonomyError
from app.engine import system2
from app.models.entities import TaxonomyVersion

log = logging.getLogger(__name__)
MAX_OPTIONS = 26          # the causal-LM prompt labels options A..Z


def taxonomy_path(settings, profile_key: str) -> Path:
    return settings.taxonomy_dir / f"{profile_key}.json"


def _name(v) -> str:
    return " ".join(str(v).split())


def clean_taxonomy(raw, settings, profile_key: str) -> tuple[dict[str, list[str]], list[str]]:
    """Normalise and validate. Returns (clean taxonomy, warnings). Hard errors raise TaxonomyError."""
    if not isinstance(raw, dict):
        raise TaxonomyError("A taxonomy must be an object of {main category: [sub-categories]}.")
    others = settings.others_label
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for main, subs in raw.items():
        name = _name(main)
        if not name:
            raise TaxonomyError("A main category has an empty name.")
        if len(name) > 80:
            raise TaxonomyError(f"Category name is too long (max 80 characters): '{name[:30]}...'")
        if name.casefold() in seen:
            raise TaxonomyError(f"Main category '{name}' appears twice.")
        seen.add(name.casefold())
        if name.casefold() == others.casefold():
            out[others] = []                                     # catch-all: sub-labels are generated per row
            continue
        if not isinstance(subs, list):
            raise TaxonomyError(f"'{name}' must contain a list of sub-categories.")
        clean, dedupe = [], set()
        for s in subs:
            n = _name(s)
            if n and n.casefold() not in dedupe:
                if len(n) > 80:
                    raise TaxonomyError(f"Sub-category name is too long (max 80 characters): '{n[:30]}...'")
                dedupe.add(n.casefold())
                clean.append(n)
        out[name] = clean
    out = {**{k: v for k, v in out.items() if k != others}, others: []}   # catch-all always last

    warnings: list[str] = []
    max_main, max_sub = settings.eff(profile_key, "max_main_categories"), settings.eff(profile_key, "max_sub_categories")
    if len(out) > MAX_OPTIONS or any(len(v) > MAX_OPTIONS for v in out.values()):
        raise TaxonomyError(f"At most {MAX_OPTIONS} options per level are supported.")
    over = []
    if len(out) > max_main:
        over.append(f"{len(out)} main categories (recommended max {max_main}, including '{others}')")
    over += [f"'{k}' has {len(v)} sub-categories (recommended max {max_sub})" for k, v in out.items() if len(v) > max_sub]
    if over:
        if settings.enforce_taxonomy_caps:
            raise TaxonomyError("Taxonomy exceeds the configured limits: " + "; ".join(over) + ".",
                                hint="Trim it, or raise max_main_categories / max_sub_categories in config.")
        warnings += [f"{o}. Small models lose accuracy with many options." for o in over]
    if len(out) == 1:
        warnings.append(f"Only '{others}' is defined, so every row will land there. Add at least one category.")
    return out, warnings


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def latest(db: Session, profile_key: str) -> TaxonomyVersion | None:
    return db.scalars(select(TaxonomyVersion).where(TaxonomyVersion.profile == profile_key)
                      .order_by(TaxonomyVersion.version.desc())).first()


def history(db: Session, profile_key: str, limit: int = 15) -> list[TaxonomyVersion]:
    return list(db.scalars(select(TaxonomyVersion).where(TaxonomyVersion.profile == profile_key)
                           .order_by(TaxonomyVersion.version.desc()).limit(limit)))


def save(db: Session, settings, profile_key: str, raw, *, source: str, note: str = "") -> tuple[TaxonomyVersion, list[str]]:
    """Validate -> write the live file -> record a version (skipped when nothing changed)."""
    clean, warnings = clean_taxonomy(raw, settings, profile_key)
    text = json.dumps(clean, indent=2, ensure_ascii=False)
    last = latest(db, profile_key)
    if last and json.loads(last.content_json) == clean:
        _atomic_write(taxonomy_path(settings, profile_key), text)
        return last, warnings
    ver = TaxonomyVersion(profile=profile_key, version=(last.version + 1 if last else 1),
                          content_json=text, source=source, note=note[:300])
    db.add(ver)
    db.commit()
    _atomic_write(taxonomy_path(settings, profile_key), text)
    log.info("Taxonomy '%s' saved as v%d (%s)", profile_key, ver.version, source)
    return ver, warnings


def load_current(db: Session, settings, profile_key: str) -> tuple[dict, int] | None:
    """The file on disk wins: a hand-edit is detected and recorded as a new version."""
    path = taxonomy_path(settings, profile_key)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TaxonomyError(f"{path.name} is not valid JSON (line {e.lineno}, column {e.colno}).",
                            hint=f"Fix the file at {path} or restore a version on the Taxonomy page.")
    clean, _ = clean_taxonomy(raw, settings, profile_key)
    last = latest(db, profile_key)
    if last is None or json.loads(last.content_json) != clean:
        last, _ = save(db, settings, profile_key, clean, source="file" if last is None else "file-edit",
                       note="Taxonomy file changed outside the app")
    return clean, last.version


def merge(existing: dict, proposed: dict, settings, profile_key: str) -> dict:
    """Keep everything the user has; add only what System 2 proposes that is new (within the caps)."""
    others = settings.others_label
    max_main, max_sub = settings.eff(profile_key, "max_main_categories"), settings.eff(profile_key, "max_sub_categories")
    out = {k: list(v) for k, v in existing.items() if k != others}
    lower = {k.casefold(): k for k in out}
    for main, subs in proposed.items():
        if main == others:
            continue
        key = lower.get(main.casefold())
        if key is None and len(out) < max_main - 1:
            out[main], lower[main.casefold()], key = [], main, main
        if key is not None:
            have = {s.casefold() for s in out[key]}
            out[key] += [s for s in subs if s.casefold() not in have][: max(0, max_sub - len(out[key]))]
    return {**out, others: []}


def bootstrap(db: Session, settings, profile_key: str, narrations: list[str]) -> tuple[dict, int]:
    """First use of a profile: System 2 designs it; if unavailable, fall back to the profile's seed."""
    profile = settings.profile(profile_key)
    failure: System2Error | None = None
    if settings.system2_enabled and settings.auto_generate_taxonomy and narrations:
        try:
            tax = system2.generate_taxonomy(narrations, settings, profile_key)
            ver, _ = save(db, settings, profile_key, tax, source="system2", note=f"Designed by {settings.system2_model}")
            return json.loads(ver.content_json), ver.version
        except System2Error as e:
            failure = e
            log.warning("System 2 unavailable for '%s': %s", profile_key, e.message)
    if profile.seed_taxonomy:
        note = "Starter taxonomy from config" + (f" (System 2 unavailable: {failure.message})" if failure else "")
        ver, _ = save(db, settings, profile_key, profile.seed_taxonomy, source="seed", note=note)
        return json.loads(ver.content_json), ver.version
    raise TaxonomyError(f"No taxonomy exists for '{profile.label}' and none could be created automatically.",
                        hint="Open the Taxonomy page and add categories, start Ollama for System 2, or define "
                             "seed_taxonomy for this profile in config.",
                        detail=failure.message if failure else "System 2 is disabled.")


def ensure(db: Session, settings, profile_key: str, narrations: list[str]) -> tuple[dict, int]:
    """What the pipeline calls: the current taxonomy for a profile, creating one on first use."""
    return load_current(db, settings, profile_key) or bootstrap(db, settings, profile_key, narrations)


def restore(db: Session, settings, profile_key: str, version_id: int) -> TaxonomyVersion:
    old = db.get(TaxonomyVersion, version_id)
    if old is None or old.profile != profile_key:
        raise TaxonomyError("That taxonomy version no longer exists.")
    ver, _ = save(db, settings, profile_key, json.loads(old.content_json), source="restore",
                  note=f"Restored from v{old.version}")
    return ver
