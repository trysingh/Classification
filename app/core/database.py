"""
database.py
-----------
SQLite engine + session factory. WAL mode lets web requests and background workers
read/write at the same time; busy_timeout makes writers wait instead of failing.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

_sf: sessionmaker | None = None


def init_db(settings) -> sessionmaker:
    """Create the engine, ensure tables exist, and return the session factory."""
    global _sf
    from app.models.entities import Base  # local import: models are optional at import time of this module

    engine = create_engine(URL.create("sqlite", database=str(settings.db_path)),
                           connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record):
        cur = conn.cursor()
        for p in ("journal_mode=WAL", "synchronous=NORMAL", "foreign_keys=ON", "busy_timeout=30000"):
            cur.execute(f"PRAGMA {p}")
        cur.close()

    Base.metadata.create_all(engine)
    _sf = sessionmaker(bind=engine, expire_on_commit=False)
    return _sf


def get_sf() -> sessionmaker:
    if _sf is None:
        raise RuntimeError("Database not initialised: call init_db(settings) first.")
    return _sf


def get_db():
    """FastAPI dependency: one short-lived session per request."""
    s: Session = get_sf()()
    try:
        yield s
    finally:
        s.close()
