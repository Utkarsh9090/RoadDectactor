"""Centralized logging module for the AI Smart Road Defect Platform.

Configures Python's built-in logging system with standardized formatting,
stream handlers, and verbosity management across all modules.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from config.settings import settings


def get_logger(
    name: str = "smart_road_detection",
    log_level: Optional[str] = None,
) -> logging.Logger:
    """Retrieves or configures a centralized logger instance.

    Args:
        name: Name of the logger, typically __name__ of the calling module.
        log_level: Optional logging level override (e.g., 'DEBUG', 'INFO', 'WARNING').

    Returns:
        logging.Logger: Configured logger with standardized formatter.
    """
    logger = logging.getLogger(name)

    # Determine logging level
    effective_level_str = log_level.upper() if log_level else settings.LOG_LEVEL.upper()
    numeric_level = getattr(logging, effective_level_str, logging.INFO)
    logger.setLevel(numeric_level)

    # Prevent duplicate handlers if logger was already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    name: str = "smart_road_detection",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger instance with an integer level.

    Args:
        name: Name of the logger.
        level: Integer logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=settings.LOG_FORMAT,
            datefmt=settings.LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
