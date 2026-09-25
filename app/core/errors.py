"""
errors.py
---------
One error vocabulary for the whole app. Raise these on purpose so every layer
(engine, services, controllers) reports problems the same way:

    message -> what happened (user-facing)
    hint    -> how to fix it
    detail  -> technical context for debugging

main.py turns them into JSON (API calls) or an error page (browser).
"""


class AppError(Exception):
    status_code = 500
    code = "app_error"

    def __init__(self, message: str, *, hint: str | None = None, detail: str | None = None):
        super().__init__(message)
        self.message, self.hint, self.detail = message, hint, detail


class ConfigError(AppError):
    status_code = 500
    code = "config_error"


class ValidationFailed(AppError):
    status_code = 400
    code = "invalid_input"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class FileReadError(AppError):
    status_code = 422
    code = "file_unreadable"


class TaxonomyError(AppError):
    status_code = 422
    code = "taxonomy_error"


class System2Error(AppError):
    status_code = 502
    code = "system2_unavailable"


class BackendError(AppError):
    status_code = 503
    code = "engine_unavailable"
