import logging
import json
from datetime import datetime, timezone
from typing import Optional
from app.core.middleware import get_request_id, RequestIdFilter
from app.core.config import settings

class StructuredJsonFormatter(logging.Formatter):
    """
    JSON log formatter outputting machine-readable JSON objects with
    timestamp, log level, logger name, request ID, and message.
    """
    def format(self, record: logging.LogRecord) -> str:
        req_id = getattr(record, "request_id", None) or get_request_id() or "-"
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": req_id,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class StandardCorrelationFormatter(logging.Formatter):
    """
    Human-readable log formatter with correlation request ID.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(
            fmt=fmt or "%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id") or not record.request_id:
            record.request_id = get_request_id() or "-"
        return super().format(record)


def setup_logging(level: str = "INFO", json_logs: Optional[bool] = None) -> None:
    """
    Configure gateway-wide logging with correlation ID injection
    and structured JSON or standard formatted output.
    """
    use_json = settings.JSON_LOGS if json_logs is None else json_logs
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicate logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.addFilter(RequestIdFilter())

    if use_json:
        console_handler.setFormatter(StructuredJsonFormatter())
    else:
        console_handler.setFormatter(StandardCorrelationFormatter())

    root_logger.addHandler(console_handler)

