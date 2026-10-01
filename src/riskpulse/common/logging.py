"""Project-wide logger (loguru) with a single configuration point."""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def get_logger(level: str = "INFO"):
    """Return the shared loguru logger, configuring stderr output once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")
        _CONFIGURED = True
    return logger
