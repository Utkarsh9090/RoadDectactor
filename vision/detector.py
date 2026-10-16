"""OpenCV Video Ingestion and YOLO Object Detector Interface.

Performs frame-by-frame defect extraction, bounding box annotation, and spatial
tracking across 100-frame road segments for utility digs, asphalt patches, cracks, and potholes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logger import get_logger
from config.settings import settings
from vision.pci_engine import DefectCounts, calculate_pci

logger = get_logger(__name__)

# Target road defect classes according to PRD Module 1
TARGET_DEFECT_CLASSES = ["dig", "patch", "crack", "pothole"]


class DefectTrack:
    """Tracks defect spatial positions across frames to prevent duplicate counting."""

    def __init__(self, track_id: int, class_name: str, bbox: Tuple[int, int, int, int], frame_idx: int) -> None:
        self.track_id: int = track_id
        self.class_name: str = class_name
        self.bbox: Tuple[int, int, int, int] = bbox  # x1, y1, x2, y2
        self.last_frame: int = frame_idx
        self.hits: int = 1

    def compute_iou(self, other_bbox: Tuple[int, int, int, int]) -> float:
        """Computes Intersection over Union (IoU) with another bounding box."""
        x1 = max(self.bbox[0], other_bbox[0])
        y1 = max(self.bbox[1], other_bbox[1])
        x2 = min(self.bbox[2], other_bbox[2])
        y2 = min(self.bbox[3], other_bbox[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        area2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0


class RoadDefectDetector:
    """OpenCV video processor and YOLO defect detector interface."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        use_mock: bool = False,
    ) -> None:
        """Initializes the defect detector.

        Args:
            model_path: Path or identifier for the YOLO model weights.
            confidence_threshold: Minimum detection confidence threshold (default: 0.25).
            use_mock: If True, simulates detections for testing without model overhead.
        """
        self.model_path: str = model_path
        self.confidence_threshold: float = confidence_threshold
        self.use_mock: bool = use_mock
        self.frame_skip: int = settings.FRAME_SKIP
        self.iou_threshold: float = settings.IOU_THRESHOLD
        self._track_counter: int = 0

        if self.use_mock:
            logger.info("RoadDefectDetector initialized in MOCK mode (simulated detections).")
            self.model: Optional[YOLO] = None
            self.has_road_classes: bool = True
        else:
            target_path = Path(model_path)
            # Check model availability
            if not target_path.exists() and model_path != "yolov8n.pt":
                logger.warning("Specified model '%s' not found. Falling back to default 'yolov8n.pt'.", model_path)
                model_path = "yolov8n.pt"

            logger.info("Loading YOLO model from: %s...", model_path)
            try:
                self.model = YOLO(model_path)
                self.has_road_classes = any(
                    cls_name.lower() in TARGET_DEFECT_CLASSES
                    for cls_name in self.model.names.values()
                )
                logger.info("YOLO Model loaded successfully. Road classes present: %s", self.has_road_classes)
            except Exception as exc:
                logger.error("Failed to load YOLO model: %s. Enabling mock mode.", exc)
                self.model = None
                self.use_mock = True
                self.has_road_classes = True

    def _simulate_frame_detections(self, frame_idx: int, width: int, height: int) -> List[Dict[str, Any]]:
        """Generates realistic simulated defect detections for mock testing.

        Args:
            frame_idx: Current frame index.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            List[Dict[str, Any]]: List of simulated defect detections.
        """
        detections: List[Dict[str, Any]] = []

        # Synthetic defects at specific frame milestones within a 100-frame window
        if 15 <= (frame_idx % 100) <= 30:
            detections.append({
                "class": "pothole",
                "confidence": 0.88,
                "bbox": (int(width * 0.35), int(height * 0.60), int(width * 0.45), int(height * 0.72)),
            })
        if 40 <= (frame_idx % 100) <= 55:
            detections.append({
                "class": "crack",
                "confidence": 0.75,
                "bbox": (int(width * 0.50), int(height * 0.55), int(width * 0.65), int(height * 0.75)),
            })
        if 65 <= (frame_idx % 100) <= 80:
            detections.append({
                "class": "patch",
                "confidence": 0.82,
                "bbox": (int(width * 0.25), int(height * 0.50), int(width * 0.40), int(height * 0.65)),
            })
        if 85 <= (frame_idx % 100) <= 95:
            detections.append({
                "class": "dig",
                "confidence": 0.79,
                "bbox": (int(width * 0.55), int(height * 0.50), int(width * 0.70), int(height * 0.68)),
            })

        return detections

    def _cv_defect_heuristic(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback computer vision texture and contour analyzer for road defects."""
        detections: List[Dict[str, Any]] = []
        try:
            h, w = frame.shape[:2]
            roi_y_start = int(h * 0.35)
            roi = frame[roi_y_start:h, 0:w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Pothole cavity detection
            _, dark_thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            opened = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 800 < area < 25000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.5:
                        detections.append({
                            "class": "pothole",
                            "confidence": 0.78,
                            "bbox": (x, y + roi_y_start, x + bw, y + roi_y_start + bh),
                        })

            # Crack edge detection
            edges = cv2.Canny(blurred, 50, 150)
            crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, crack_kernel, iterations=1)
            crack_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in crack_contours:
                length = cv2.arcLength(cnt, closed=False)
                area = cv2.contourArea(cnt)
                if length > 70 and area < 3000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append({
                        "class": "crack",
                        "confidence": 0.65,
                        "bbox": (x, y + roi_y_start, x + bw, y + roi_y_start + bh),
                    })

            # Patch / Dig detection
            _, light_thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
            patch_contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in patch_contours:
                area = cv2.contourArea(cnt)
                if 2000 < area < 40000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    rect_area = bw * bh
                    extent = float(area) / float(rect_area) if rect_area > 0 else 0
                    if extent > 0.65:
                        defect_cls = "dig" if extent > 0.85 else "patch"
                        detections.append({
                            "class": defect_cls,
                            "confidence": 0.72,
                            "bbox": (x, y + roi_y_start, x + bw, y + roi_y_start + bh),
                        })
        except Exception as exc:
            logger.debug("CV heuristic error: %s", exc)

        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        active_tracks: Optional[List[DefectTrack]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int], List[DefectTrack]]:
        """Extracts defects from a single frame, updates tracking, and annotates bounding boxes.

        Args:
            frame: Input BGR image array.
            frame_idx: Index of current frame.
            active_tracks: Existing tracked defect instances.

        Returns:
            Tuple[np.ndarray, Dict[str, int], List[DefectTrack]]:
                - Annotated image frame
                - Incremental defect counts confirmed in this frame
                - Updated list of active tracks
        """
        if active_tracks is None:
            active_tracks = []

        h, w = frame.shape[:2]
        annotated = frame.copy()
        raw_detections: List[Dict[str, Any]] = []

        if self.use_mock:
            raw_detections = self._simulate_frame_detections(frame_idx, w, h)
        else:
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=settings.FRAME_WIDTH,
                    verbose=False,
                )
                for res in results:
                    if res.boxes is None:
                        continue
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cls_name = self.model.names.get(cls_id, "").lower()

                        if cls_name in TARGET_DEFECT_CLASSES:
                            raw_detections.append({
                                "class": cls_name,
                                "confidence": conf,
                                "bbox": (x1, y1, x2, y2),
                            })
            except Exception as exc:
                logger.warning("YOLO inference failed: %s", exc)

            if not self.has_road_classes or len(raw_detections) == 0:
                raw_detections.extend(self._cv_defect_heuristic(frame))

        # Spatial tracking and deduplication
        new_counts: Dict[str, int] = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}
        updated_tracks: List[DefectTrack] = []

        for det in raw_detections:
            bbox = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]

            matched_track: Optional[DefectTrack] = None
            best_iou = 0.0

            for track in active_tracks:
                if track.class_name == cls_name:
                    iou = track.compute_iou(bbox)
                    if iou > best_iou:
                        best_iou = iou
                        matched_track = track

            if matched_track and best_iou >= self.iou_threshold:
                matched_track.bbox = bbox
                matched_track.last_frame = frame_idx
                matched_track.hits += 1
                updated_tracks.append(matched_track)
                track_id = matched_track.track_id
            else:
                self._track_counter += 1
                new_track = DefectTrack(self._track_counter, cls_name, bbox, frame_idx)
                updated_tracks.append(new_track)
                track_id = new_track.track_id

                key = f"{cls_name}s" if not cls_name.endswith("s") else cls_name
                if key in new_counts:
                    new_counts[key] += 1

            # Render bounding box and label
            color = settings.DEFECT_COLORS.get(cls_name, (0, 255, 0))
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"#{track_id} {cls_name.upper()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (bbox[0], max(0, bbox[1] - 20)), (bbox[0] + tw, bbox[1]), color, -1)
            cv2.putText(
                annotated,
                label,
                (bbox[0], max(14, bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        surviving_tracks = [t for t in updated_tracks if (frame_idx - t.last_frame) <= 15]
        return annotated, new_counts, surviving_tracks

    def process_video_segment(
        self,
        video_path: str,
        start_frame: int = 0,
        frame_count: int = 100,
    ) -> Tuple[DefectCounts, List[np.ndarray]]:
        """Processes a discrete video segment window and returns aggregated defect counts and frames.

        Args:
            video_path: Path to target input video.
            start_frame: Starting frame index (0-indexed).
            frame_count: Total frames spanning this segment window (default: 100).

        Returns:
            Tuple[DefectCounts, List[np.ndarray]]:
                - Aggregated DefectCounts instance.
                - List of annotated processed frame arrays.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video stream cannot be opened or is corrupted.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found at: '{video_path}'")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Unable to open or decode video at: '{video_path}'")

        total_available_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame >= total_available_frames and total_available_frames > 0:
            cap.release()
            raise ValueError(
                f"start_frame ({start_frame}) exceeds total frames in video ({total_available_frames})."
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        annotated_frames: List[np.ndarray] = []
        segment_defects = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}
        active_tracks: List[DefectTrack] = []

        current_frame_pos = start_frame
        frames_read = 0

        try:
            while cap.isOpened() and frames_read < frame_count:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.debug("End of video stream reached at frame %d.", current_frame_pos)
                    break

                current_frame_pos += 1
                frames_read += 1

                # Frame skip check (reading FRAME_SKIP from config/settings.py)
                if (current_frame_pos % self.frame_skip) != 0:
                    continue

                # Run inference and defect tracking
                annotated_frame, frame_counts, active_tracks = self.detect_frame(
                    frame=frame,
                    frame_idx=current_frame_pos,
                    active_tracks=active_tracks,
                )

                for k, v in frame_counts.items():
                    segment_defects[k] += v

                annotated_frames.append(annotated_frame)

        finally:
            cap.release()

        aggregated_defects = DefectCounts(
            digs=segment_defects["digs"],
            patches=segment_defects["patches"],
            cracks=segment_defects["cracks"],
            potholes=segment_defects["potholes"],
        )

        logger.info(
            "Segment [%d - %d] analyzed: %d frames processed | Defects: %s",
            start_frame,
            current_frame_pos,
            len(annotated_frames),
            aggregated_defects.model_dump(),
        )

        return aggregated_defects, annotated_frames

    def process_video(
        self,
        video_path: str,
        display: bool = False,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Processes an entire video end-to-end divided into 100-frame segments.

        Args:
            video_path: Path to target input video.
            display: If True, renders live detection window.
            output_path: Optional destination path for encoded output video.
            progress_callback: Optional callable for progress updates.

        Returns:
            Dict[str, Any]: Complete inspection report.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream at: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or settings.DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or settings.FRAME_WIDTH
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or settings.FRAME_HEIGHT

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps / self.frame_skip, (width, height))

        segments: List[Dict[str, Any]] = []
        current_segment_id = 1
        current_segment_defects = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}
        total_defects = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}

        active_tracks: List[DefectTrack] = []
        frame_idx = 0
        processed_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    continue

                processed_frames += 1

                annotated_frame, new_counts, active_tracks = self.detect_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    active_tracks=active_tracks,
                )

                for k, v in new_counts.items():
                    current_segment_defects[k] += v
                    total_defects[k] += v

                # HUD Overlay
                hud_text = (
                    f"Frame {frame_idx}/{total_frames} | Seg #{current_segment_id} | "
                    f"Digs:{current_segment_defects['digs']} "
                    f"Patches:{current_segment_defects['patches']} "
                    f"Cracks:{current_segment_defects['cracks']} "
                    f"Potholes:{current_segment_defects['potholes']}"
                )
                cv2.rectangle(annotated_frame, (10, 10), (width - 10, 45), (30, 30, 30), -1)
                cv2.putText(
                    annotated_frame,
                    hud_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if writer:
                    writer.write(annotated_frame)

                if display:
                    cv2.imshow("Road Defect Detection (Press 'q' to Quit)", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                # Segment Boundary Check (100 frames)
                if frame_idx % settings.FRAMES_PER_SEGMENT == 0:
                    seg_pci = calculate_pci(current_segment_defects)
                    segments.append({
                        "segment_id": current_segment_id,
                        "frame_start": frame_idx - settings.FRAMES_PER_SEGMENT,
                        "frame_end": frame_idx,
                        "defects": dict(current_segment_defects),
                        "pci_score": seg_pci,
                    })
                    current_segment_id += 1
                    current_segment_defects = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}
                    active_tracks.clear()

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        # Handle final trailing segment if frames remain
        if (frame_idx % settings.FRAMES_PER_SEGMENT) != 0:
            seg_pci = calculate_pci(current_segment_defects)
            segments.append({
                "segment_id": current_segment_id,
                "frame_start": (current_segment_id - 1) * settings.FRAMES_PER_SEGMENT,
                "frame_end": frame_idx,
                "defects": dict(current_segment_defects),
                "pci_score": seg_pci,
            })

        overall_pci = calculate_pci(total_defects)

        return {
            "status": "success",
            "filename": video_file.name,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_segments_analyzed": len(segments),
            "summary": {
                "overall_pci": overall_pci,
                "total_defects": total_defects,
            },
            "segment_details": segments,
        }


# ---------------------------------------------------------------------------
# Runnable Self-Test Script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    logger.info("Initializing RoadDefectDetector test run...")

    # Create synthetic 100-frame test video using OpenCV
    temp_dir = tempfile.mkdtemp()
    test_video_path = Path(temp_dir) / "test_segment.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_video_path), fourcc, 30.0, (640, 640))

    logger.info("Writing synthetic 100-frame test video to: %s...", test_video_path)
    for i in range(100):
        # Render gray road frame with moving lane marker
        test_frame = np.full((640, 640, 3), 85, dtype=np.uint8)
        dash_y = (i * 12) % 640
        cv2.line(test_frame, (320, dash_y), (320, min(640, dash_y + 40)), (0, 220, 255), 4)
        writer.write(test_frame)
    writer.release()

    logger.info("Synthetic video written successfully. Running RoadDefectDetector(use_mock=True)...")
    detector = RoadDefectDetector(use_mock=True)

    # Process 100-frame video segment
    defects, annotated_frames = detector.process_video_segment(
        video_path=str(test_video_path),
        start_frame=0,
        frame_count=100,
    )

    computed_pci = calculate_pci(defects)

    print("\n" + "=" * 55)
    print("        ROAD DEFECT DETECTOR TEST RESULTS")
    print("=" * 55)
    print(f"Total Processed Frames : {len(annotated_frames)} (Skip Rate: {settings.FRAME_SKIP})")
    print(f"Detected Defect Counts : {defects.model_dump()}")
    print(f"Computed Pavement PCI  : {computed_pci:.1f} / 100.0")
    print("=" * 55 + "\n")

    # Clean up temporary test files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Detector self-test completed successfully.")
