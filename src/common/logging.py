"""Structured logging setup."""

from __future__ import annotations

import logging
from typing import Any

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    if kwargs:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        logger.info("%s | %s", message, parts)
    else:
        logger.info(message)
