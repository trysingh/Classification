"""
logging_setup.py
----------------
Console + rotating file logs, every line stamped with the request id so a user-reported
"Reference: ab12cd34ef" can be grepped straight to the failing request in data/logs/app.log.
"""
import contextvars
import logging
import sys
from logging.handlers import RotatingFileHandler

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(settings) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    for h in list(root.handlers):                       # idempotent: safe when the app is re-created (tests, reload)
        if getattr(h, "_openjev", False):
            root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(settings.log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
    ]
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(_RequestIdFilter())
        h._openjev = True
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
