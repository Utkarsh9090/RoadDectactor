"""Unit tests for configuration settings and centralized logger."""

from __future__ import annotations

import logging
from pathlib import Path
import pytest
from config.logger import get_logger, setup_logging
from config.settings import Settings, settings


def test_default_settings_values() -> None:
    """Verifies that default configuration values adhere to PRD and Stage 1 specs."""
    assert settings.FRAME_SKIP == 5
    assert settings.DEFAULT_FPS == 30
    assert settings.DIG_PENALTY == 12.0
    assert settings.PATCH_PENALTY == 8.0
    assert settings.CRACK_PENALTY == 4.0
    assert settings.POTHOLE_PENALTY == 15.0

    assert settings.VIDEO_INPUT_DIR.exists()
    assert settings.MODEL_SAVE_DIR.exists()
    assert str(settings.VIDEO_INPUT_DIR) == settings.VIDEO_INPUT_PATH
    assert settings.PCI_WEIGHTS["dig"] == 12.0
    assert settings.PCI_WEIGHTS["patch"] == 8.0
    assert settings.PCI_WEIGHTS["crack"] == 4.0
    assert settings.PCI_WEIGHTS["pothole"] == 15.0
    assert settings.LOG_LEVEL == "INFO"
    assert "%(asctime)s" in settings.LOG_FORMAT


def test_directory_validation_failure() -> None:
    """Settings must raise FileNotFoundError if a required operational directory is missing."""
    test_settings = Settings()
    test_settings.VIDEO_INPUT_DIR = Path("non_existent_test_directory_404")

    with pytest.raises(FileNotFoundError) as exc_info:
        test_settings.validate_directories(create_if_missing=False)

    assert "Required directory 'VIDEO_INPUT_DIR' does not exist" in str(exc_info.value)


def test_centralized_logger() -> None:
    """Tests logger initialization, handler attachment, and formatting."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
