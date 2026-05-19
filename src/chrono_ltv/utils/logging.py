"""Centralised logging configuration using Loguru."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    serialize: bool = False,
) -> None:
    """Configure Loguru for the entire application.

    Call once at process startup. Subsequent calls are idempotent because
    the default sink is removed before adding new ones.
    """
    logger.remove()  # drop the default stderr sink

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=fmt,
            level=level,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            backtrace=True,
            diagnose=True,
            enqueue=True,  # thread-safe async logging
        )


def get_logger(name: str) -> Any:
    """Return a child logger bound to *name* (module path)."""
    return logger.bind(name=name)
