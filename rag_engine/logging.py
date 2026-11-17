"""Application logging with a request id context."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

from rag_engine.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("rag_engine")
    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL.upper())
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.LOG_LEVEL.upper())
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def set_request_id(value: str | None) -> None:
    request_id_var.set(value)


def clear_request_id() -> None:
    request_id_var.set(None)


logger = setup_logging()
