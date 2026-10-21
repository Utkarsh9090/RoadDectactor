"""Unit tests for the OpenCV video processor and YOLO defect detector interface."""

from __future__ import annotations

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from config.settings import settings
from vision.detector import RoadDefectDetector
from vision.pci_engine import DefectCounts, calculate_pci


@pytest.fixture
def synthetic_video_path() -> str:
    """Creates a temporary 100-frame synthetic test video."""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "test_100_frames.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 640))

    for i in range(100):
        frame = np.full((640, 640, 3), 70, dtype=np.uint8)
        # Moving yellow center stripe
        dash_y = (i * 10) % 640
        cv2.line(frame, (320, dash_y), (320, min(640, dash_y + 35)), (0, 220, 255), 4)
        writer.write(frame)
    writer.release()

    yield str(video_path)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_detector_initialization() -> None:
    """Verifies that detector initializes properly in both mock and real mode."""
    mock_detector = RoadDefectDetector(use_mock=True)
    assert mock_detector.use_mock is True
    assert mock_detector.model is None

    real_detector = RoadDefectDetector(model_path="yolov8n.pt", use_mock=False)
    assert real_detector.confidence_threshold == 0.25


def test_process_video_segment_mock(synthetic_video_path: str) -> None:
    """Tests process_video_segment using mock detector on a 100-frame video."""
    detector = RoadDefectDetector(use_mock=True)
    defects, annotated_frames = detector.process_video_segment(
        video_path=synthetic_video_path,
        start_frame=0,
        frame_count=100,
    )

    assert isinstance(defects, DefectCounts)
    assert defects.digs >= 0
    assert defects.patches >= 0
    assert defects.cracks >= 0
    assert defects.potholes >= 0

    # 100 frames / FRAME_SKIP (5) = 20 processed frames
    expected_frames = 100 // settings.FRAME_SKIP
    assert len(annotated_frames) == expected_frames
    assert isinstance(annotated_frames[0], np.ndarray)
    assert annotated_frames[0].shape == (640, 640, 3)

    # Verify PCI calculation on detected defects
    pci = calculate_pci(defects)
    assert 0.0 <= pci <= 100.0


def test_process_video_missing_file_raises_error() -> None:
    """Detector must raise FileNotFoundError for nonexistent video files."""
    detector = RoadDefectDetector(use_mock=True)
    with pytest.raises(FileNotFoundError):
        detector.process_video_segment(video_path="non_existent_file_404.mp4")


def test_process_video_invalid_start_frame(synthetic_video_path: str) -> None:
    """Detector must raise ValueError if start_frame exceeds video duration."""
    detector = RoadDefectDetector(use_mock=True)
    with pytest.raises(ValueError):
        detector.process_video_segment(
            video_path=synthetic_video_path,
            start_frame=500,  # Video only has 100 frames
            frame_count=100,
        )


def test_clear_perfect_road_zero_defects() -> None:
    """Ensures a clean highway with painted lane markings produces 0 false defects and 100 PCI."""
    temp_dir = tempfile.mkdtemp()
    clean_video = Path(temp_dir) / "clean_highway_test.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clean_video), fourcc, 30.0, (640, 480))

    for i in range(60):
        frame = np.full((480, 640, 3), 110, dtype=np.uint8)
        # Moving center dashed white line
        dash_y = (i * 15) % 80
        for y in range(-80 + dash_y, 480, 80):
            cv2.line(frame, (320, y), (320, y + 40), (255, 255, 255), 6)
        # Yellow and white road shoulders
        cv2.line(frame, (80, 0), (80, 480), (0, 220, 255), 4)
        cv2.line(frame, (560, 0), (560, 480), (255, 255, 255), 4)
        writer.write(frame)
    writer.release()

    try:
        detector = RoadDefectDetector(use_mock=False)
        report = detector.process_video(str(clean_video))

        assert report["summary"]["total_defects"] == {
            "digs": 0,
            "patches": 0,
            "cracks": 0,
            "potholes": 0,
        }
        assert report["summary"]["overall_pci"] == 100.0
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

