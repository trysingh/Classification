"""
main.py
-------
Application factory. Wires config -> database -> engine registry -> job manager -> routers,
and owns cross-cutting concerns: request ids, access logging and ONE consistent error contract
(JSON for API calls, an error page for browsers; both carry a reference id found in the logs).
"""
from __future__ import annotations

import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import AppSettings, settings as default_settings
from app.controllers import batch, dashboard, files, jobs, live, single, suppliers, system, taxonomy
from app.core.database import init_db
from app.core.errors import AppError
from app.core.logging_setup import request_id_var, setup_logging
from app.core.templating import APP_DIR, render, templates
from app.engine.backends import EngineRegistry
from app.models.entities import FileRun
from app.services.diagnostics_service import record_error
from app.services.job_manager import JobManager

log = logging.getLogger("openjev")
QUIET_PREFIXES = ("/static", "/api/jobs")        # polled constantly: logged at DEBUG only


def _wants_json(request: Request) -> bool:
    return request.url.path.startswith("/api/") or "application/json" in request.headers.get("accept", "")


def _fail(request: Request, status: int, message: str, *, hint=None, detail=None, code="error"):
    """The single error contract: same fields for JSON and HTML."""
    rid = getattr(request.state, "request_id", "-")
    if _wants_json(request):
        err = {"code": code, "message": message, "hint": hint, "request_id": rid}
        if detail:
            err["detail"] = detail
        return JSONResponse({"error": err}, status_code=status, headers={"X-Request-ID": rid})
    return render(request, "error.html", status_code=status, status=status, message=message, hint=hint,
                  detail=detail, request_id=rid)


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = uuid.uuid4().hex[:10]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            level = logging.DEBUG if request.url.path.startswith(QUIET_PREFIXES) else logging.INFO
            log.log(level, "%s %s %.0fms", request.method, request.url.path, (time.perf_counter() - t0) * 1000)
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        (log.error if exc.status_code >= 500 else log.warning)("%s: %s | %s", exc.code, exc.message, exc.detail or "")
        return _fail(request, exc.status_code, exc.message, hint=exc.hint, detail=exc.detail, code=exc.code)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        missing = exc.status_code == 404
        return _fail(request, exc.status_code, "Page not found." if missing else str(exc.detail),
                     hint="Use the menu to get back on track." if missing else None, code="http_error")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = "; ".join(f"{'.'.join(map(str, e['loc'][1:])) or e['loc'][0]}: {e['msg']}" for e in exc.errors())
        return _fail(request, 422, "Some inputs are not valid.", hint="Correct them and try again.",
                     detail=fields, code="validation_error")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        trace = "".join(traceback.format_exception(exc))
        log.error("Unhandled error on %s %s\n%s", request.method, request.url.path, trace)
        record_error(f"http:{request.method} {request.url.path}", f"{type(exc).__name__}: {exc}", trace, ref=rid)
        return _fail(request, 500, "Something went wrong on the server.",
                     hint=f"Quote reference {rid}; the full trace is on the Diagnostics page.",
                     detail=trace if request.app.state.settings.debug else None, code="server_error")


def _cleanup_uploads(settings: AppSettings, sf) -> None:
    """Drop abandoned previews: uploads no run refers to, older than the retention window."""
    cutoff = time.time() - settings.upload_retention_days * 86400
    try:
        with sf() as s:
            used = set(s.scalars(select(FileRun.source_path)))
        for p in settings.upload_dir.iterdir():
            if p.is_file() and p.stat().st_mtime < cutoff and str(p) not in used:
                p.unlink(missing_ok=True)
    except Exception:
        log.exception("Upload cleanup skipped")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or default_settings
    setup_logging(settings)
    sf = init_db(settings)
    registry = EngineRegistry(settings)
    job_manager = JobManager(settings, sf, registry)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _cleanup_uploads(settings, sf)
        job_manager.start()                                      # also resumes jobs interrupted by a restart
        log.info("%s ready (engine=%s, data=%s)", settings.app_name, settings.classifier_backend, settings.data_dir)
        yield
        job_manager.stop()

    app = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.settings, app.state.registry, app.state.jobs = settings, registry, job_manager
    templates.env.globals.update(settings=settings, app_name=settings.app_name, others_label=settings.others_label)
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    _install_middleware(app)
    _install_error_handlers(app)
    for module in (dashboard, single, batch, jobs, files, taxonomy, suppliers, live, system):
        app.include_router(module.router)
    return app
