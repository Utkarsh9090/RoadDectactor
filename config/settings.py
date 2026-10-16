"""Centralized configuration and settings for the AI Smart Road Defect Platform.

Defines video ingestion parameters, PCI deduction penalties, system directory paths,
and logging formats with explicit boundary checks and directory validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Final, Tuple
import numpy as np


@dataclass(frozen=False)
class Settings:
    """Application settings and configuration parameters.

    Attributes:
        FRAME_SKIP: Interval for frame extraction (process every Nth frame).
        DEFAULT_FPS: Baseline frames per second for video processing.
        DIG_PENALTY: PCI deduction weight for unsealed utility cuts.
        PATCH_PENALTY: PCI deduction weight for deteriorated asphalt patches.
        CRACK_PENALTY: PCI deduction weight for cracks (longitudinal/transverse/alligator).
        POTHOLE_PENALTY: PCI deduction weight for structural cavities.
        BASE_DIR: Project root directory.
        DATA_DIR: Root data directory.
        VIDEO_INPUT_DIR: Directory for input inspection video feeds.
        MODEL_SAVE_DIR: Directory for saved YOLO and RL model weights.
        OUTPUT_DIR: Directory for generated inspection reports and artifacts.
        LOG_LEVEL: Logging verbosity level.
        LOG_FORMAT: Standard log message formatting string.
        LOG_DATE_FORMAT: Datetime format string for log records.
    """

    # Video Processing Settings
    FRAME_SKIP: int = 5
    DEFAULT_FPS: int = 30
    FRAMES_PER_SEGMENT: int = 100
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 640
    CONFIDENCE_THRESHOLD: float = 0.40
    IOU_THRESHOLD: float = 0.45

    # PCI Deduction Penalties (PRD Section 4)
    DIG_PENALTY: float = 12.0
    PATCH_PENALTY: float = 8.0
    CRACK_PENALTY: float = 4.0
    POTHOLE_PENALTY: float = 15.0

    MAX_PCI: float = 100.0
    MIN_PCI: float = 0.0

    # Directory Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(init=False)
    VIDEO_INPUT_DIR: Path = field(init=False)
    MODEL_SAVE_DIR: Path = field(init=False)
    OUTPUT_DIR: Path = field(init=False)

    # Model Paths
    YOLO_MODEL_PATH: str = field(init=False)
    RL_MODEL_PATH: str = field(init=False)
    DEFAULT_FALLBACK_YOLO: str = "yolov8n.pt"

    # Logging Constants
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Reinforcement Learning Settings
    RL_STATE_DIM: int = 5
    RL_FAILURE_PENALTY: float = 50.0
    RL_CRITICAL_PCI_THRESHOLD: float = 40.0
    RL_PREVENTIVE_PCI_BOOST: float = 20.0
    RL_MAX_EPISODE_AGE: float = 15.0

    def __post_init__(self) -> None:
        """Initializes derived paths from environment variables and validates directories."""
        self.DATA_DIR = self.BASE_DIR / "data"

        # Allow environment overrides for directory paths
        env_video_dir = os.getenv("VIDEO_INPUT_DIR")
        self.VIDEO_INPUT_DIR = Path(env_video_dir) if env_video_dir else self.DATA_DIR / "videos"

        env_model_dir = os.getenv("MODEL_SAVE_DIR")
        self.MODEL_SAVE_DIR = Path(env_model_dir) if env_model_dir else self.DATA_DIR / "models"

        self.OUTPUT_DIR = self.DATA_DIR / "outputs"

        # Model file paths
        self.YOLO_MODEL_PATH = os.getenv(
            "YOLO_MODEL_PATH", str(self.MODEL_SAVE_DIR / "yolov8_road.pt")
        )
        self.RL_MODEL_PATH = os.getenv(
            "RL_MODEL_PATH", str(self.MODEL_SAVE_DIR / "dqn_road_policy.zip")
        )

        # Enforce directory existence validation
        self.validate_directories(create_if_missing=False)

    def validate_directories(self, create_if_missing: bool = False) -> None:
        """Validates that all required operational directories exist on the filesystem.

        Args:
            create_if_missing: If True, creates missing directories; otherwise raises FileNotFoundError.

        Raises:
            FileNotFoundError: If a required directory does not exist and create_if_missing is False.
        """
        required_dirs = [
            ("VIDEO_INPUT_DIR", self.VIDEO_INPUT_DIR),
            ("MODEL_SAVE_DIR", self.MODEL_SAVE_DIR),
        ]

        for name, dir_path in required_dirs:
            if not dir_path.exists():
                if create_if_missing:
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    raise FileNotFoundError(
                        f"Required directory '{name}' does not exist at: '{dir_path.resolve()}'. "
                        "Please ensure the directory is created before initializing settings."
                    )

    @property
    def VIDEO_INPUT_PATH(self) -> str:
        """Returns the default video input path as a string."""
        return str(self.VIDEO_INPUT_DIR)

    @property
    def PCI_WEIGHTS(self) -> Dict[str, float]:
        """Returns the dictionary mapping defect class names to their deduct values."""
        return {
            "dig": self.DIG_PENALTY,
            "patch": self.PATCH_PENALTY,
            "crack": self.CRACK_PENALTY,
            "pothole": self.POTHOLE_PENALTY,
        }

    @property
    def DEFECT_CLASSES(self) -> Dict[int, str]:
        """Returns defect class ID to name mapping."""
        return {
            0: "dig",
            1: "patch",
            2: "crack",
            3: "pothole",
        }

    @property
    def CLASS_NAME_TO_ID(self) -> Dict[str, int]:
        """Returns defect class name to ID mapping."""
        return {v: k for k, v in self.DEFECT_CLASSES.items()}

    @property
    def DEFECT_COLORS(self) -> Dict[str, Tuple[int, int, int]]:
        """Returns BGR color tuples for bounding box rendering."""
        return {
            "dig": (0, 165, 255),      # Orange
            "patch": (0, 255, 255),    # Yellow
            "crack": (255, 0, 0),      # Blue
            "pothole": (0, 0, 255),    # Red
        }

    @property
    def RL_ACTION_NAMES(self) -> Dict[int, str]:
        """Returns mapping of RL action index to human-readable names."""
        return {
            0: "Do Nothing",
            1: "Preventive Repair",
            2: "Major Resurfacing",
        }

    @property
    def RL_ACTION_COSTS(self) -> Dict[int, float]:
        """Returns mapping of RL action index to cost deduction."""
        return {
            0: 0.0,
            1: 15.0,
            2: 60.0,
        }

    @property
    def RL_OBS_LOW(self) -> np.ndarray:
        """Returns lower bound for Gymnasium observation space."""
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def RL_OBS_HIGH(self) -> np.ndarray:
        """Returns upper bound for Gymnasium observation space."""
        return np.array([100.0, 50.0, 50.0, 50.0, 30.0], dtype=np.float32)


# Global settings singleton instance
settings = Settings()
