from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextvars import ContextVar
import uuid
import logging
from typing import Optional

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")

def get_request_id() -> str:
    """Retrieve the current request ID from context."""
    return request_id_ctx_var.get()

class RequestIdFilter(logging.Filter):
    """Logging filter to inject request_id into log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts or generates a correlation ID for every request,
    stores it in contextvars for logging, and returns it in the X-Request-ID response header.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx_var.set(req_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx_var.reset(token)

