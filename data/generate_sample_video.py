"""Utility script to generate a realistic synthetic road inspection video for testing.

Renders 200 video frames simulating a road with asphalt texture, lane markings,
and defects (potholes, cracks, patches, utility cuts).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import cv2
import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SAMPLE_VIDEOS_DIR,
    setup_logging,
)

logger = setup_logging(__name__)


def generate_sample_road_video(
    output_path: Optional[str] = None,
    total_frames: int = 200,
    fps: int = 30,
) -> str:
    """Generates an MP4 test video with synthetic road defects.

    Args:
        output_path: Path where the video will be saved.
        total_frames: Number of frames to synthesize (default: 200 = 2 segments).
        fps: Frames per second.

    Returns:
        str: Absolute path of generated video file.
    """
    if output_path is None:
        target_path = SAMPLE_VIDEOS_DIR / "sample_road.mp4"
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target_path), fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    logger.info("Generating synthetic road video: %s (%d frames)...", target_path, total_frames)

    np.random.seed(42)

    # Define simulated road defects with vertical positions (y-axis)
    # Segment 1 (frames 0 - 100): 1 pothole, 2 cracks, 1 patch
    # Segment 2 (frames 101 - 200): 1 dig, 2 patches, 1 crack
    defect_events = [
        {"start_frame": 15, "type": "pothole", "x": 220, "size": 35},
        {"start_frame": 40, "type": "crack", "x": 380, "size": 80},
        {"start_frame": 65, "type": "patch", "x": 180, "size": 90},
        {"start_frame": 85, "type": "crack", "x": 310, "size": 70},
        {"start_frame": 115, "type": "dig", "x": 340, "size": 100},
        {"start_frame": 140, "type": "patch", "x": 200, "size": 85},
        {"start_frame": 165, "type": "crack", "x": 280, "size": 75},
        {"start_frame": 185, "type": "patch", "x": 360, "size": 70},
    ]

    for frame_idx in range(total_frames):
        # Base asphalt background (dark gray with texture)
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)
        # Add texture noise
        noise = np.random.randint(-12, 12, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw road boundaries (solid yellow/white lines)
        cv2.line(frame, (100, 0), (100, FRAME_HEIGHT), (255, 255, 255), 4)
        cv2.line(frame, (FRAME_WIDTH - 100, 0), (FRAME_WIDTH - 100, FRAME_HEIGHT), (255, 255, 255), 4)

        # Draw moving dashed centerline
        dash_offset = (frame_idx * 14) % 80
        for y in range(-80 + dash_offset, FRAME_HEIGHT + 80, 80):
            cv2.line(frame, (FRAME_WIDTH // 2, y), (FRAME_WIDTH // 2, y + 45), (0, 220, 255), 4)

        # Render active defects for this frame
        for event in defect_events:
            event_start = event["start_frame"]
            if event_start <= frame_idx < event_start + 25:
                progress = (frame_idx - event_start) / 25.0
                curr_y = int(220 + (progress * 380))
                curr_x = event["x"]
                sz = event["size"]

                if event["type"] == "pothole":
                    # Dark structural cavity
                    cv2.ellipse(frame, (curr_x, curr_y), (sz, int(sz * 0.7)), 0, 0, 360, (25, 25, 25), -1)
                    cv2.ellipse(frame, (curr_x - 3, curr_y - 2), (int(sz * 0.7), int(sz * 0.4)), 0, 0, 360, (10, 10, 10), -1)
                elif event["type"] == "crack":
                    # Longitudinal/transverse crack line
                    pts = np.array([
                        [curr_x, curr_y - sz // 2],
                        [curr_x + 10, curr_y - sz // 4],
                        [curr_x - 8, curr_y + sz // 4],
                        [curr_x + 5, curr_y + sz // 2],
                    ], np.int32)
                    cv2.polylines(frame, [pts], isClosed=False, color=(30, 30, 30), thickness=3)
                elif event["type"] == "patch":
                    # Deteriorated asphalt rectangular patch
                    cv2.rectangle(
                        frame,
                        (curr_x - sz // 2, curr_y - sz // 3),
                        (curr_x + sz // 2, curr_y + sz // 3),
                        (160, 160, 160),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (curr_x - sz // 2, curr_y - sz // 3),
                        (curr_x + sz // 2, curr_y + sz // 3),
                        (50, 50, 50),
                        2,
                    )
                elif event["type"] == "dig":
                    # Unsealed utility cut / trench
                    cv2.rectangle(
                        frame,
                        (curr_x - sz // 2, curr_y - sz // 4),
                        (curr_x + sz // 2, curr_y + sz // 4),
                        (170, 170, 170),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (curr_x - sz // 2 + 5, curr_y - sz // 4 + 4),
                        (curr_x + sz // 2 - 5, curr_y + sz // 4 - 4),
                        (40, 40, 40),
                        -1,
                    )

        writer.write(frame)

    writer.release()
    logger.info("Video successfully generated at: %s", target_path)
    return str(target_path)


if __name__ == "__main__":
    generate_sample_road_video()
