"""OpenCV Video Ingestion and YOLO Object Detector Interface.

Performs frame-by-frame defect extraction, bounding box annotation, and spatial
tracking across 100-frame road segments for utility digs, asphalt patches, cracks, and potholes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
TARGET_DEFECT_CLASSES: List[str] = ["dig", "patch", "crack", "pothole"]

# Mapping of class aliases (fine-tuned models, RDD2022, Roboflow labels) to canonical classes
CLASS_ALIASES: Dict[str, str] = {
    "d00": "crack",
    "d10": "crack",
    "d20": "crack",
    "longitudinal_crack": "crack",
    "transverse_crack": "crack",
    "alligator_crack": "crack",
    "crack": "crack",
    "cracks": "crack",
    "d40": "pothole",
    "pothole": "pothole",
    "potholes": "pothole",
    "utility_cut": "dig",
    "utility_cuts": "dig",
    "utility_dig": "dig",
    "utility_digs": "dig",
    "dig": "dig",
    "digs": "dig",
    "repair_patch": "patch",
    "repair_patches": "patch",
    "patch": "patch",
    "patches": "patch",
}

# Standard COCO dynamic obstacles to mask from road pavement analysis
COCO_OBSTACLE_CLASSES: Set[str] = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


class DefectTrack:
    """Tracks defect spatial positions across frames to prevent duplicate counting."""

    def __init__(self, track_id: int, class_name: str, bbox: Tuple[int, int, int, int], frame_idx: int) -> None:
        self.track_id: int = track_id
        self.class_name: str = class_name
        self.bbox: Tuple[int, int, int, int] = bbox  # x1, y1, x2, y2
        self.last_frame: int = frame_idx
        self.hits: int = 1
        self.counted: bool = False

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

    def is_match(self, other_bbox: Tuple[int, int, int, int], iou_thresh: Optional[float] = None) -> bool:
        """Determines if other_bbox corresponds to the same physical defect instance.

        Combines standard IoU with forward vehicle camera displacement tracking.
        """
        thresh = iou_thresh if iou_thresh is not None else settings.TRACKER_IOU
        iou = self.compute_iou(other_bbox)
        if iou >= thresh:
            return True

        # Centroid distance & forward camera motion check
        c1_x = (self.bbox[0] + self.bbox[2]) / 2.0
        c1_y = (self.bbox[1] + self.bbox[3]) / 2.0
        c2_x = (other_bbox[0] + other_bbox[2]) / 2.0
        c2_y = (other_bbox[1] + other_bbox[3]) / 2.0

        dx = abs(c2_x - c1_x)
        dy = c2_y - c1_y  # Forward vehicle motion moves road objects downwards in camera view

        w1 = self.bbox[2] - self.bbox[0]
        h1 = self.bbox[3] - self.bbox[1]
        w2 = other_bbox[2] - other_bbox[0]
        h2 = other_bbox[3] - other_bbox[1]

        # Trajectory tolerances
        x_tol = max(45.0, 0.6 * max(w1, w2))
        y_tol_forward = max(110.0, 1.8 * max(h1, h2))

        if dx <= x_tol and -15.0 <= dy <= y_tol_forward:
            area1 = w1 * h1
            area2 = w2 * h2
            area_ratio = min(area1, area2) / max(1.0, max(area1, area2))
            if area_ratio >= 0.20:
                return True

        return False


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
            logger.warning(
                "ROAD DEFECT DETECTOR INITIALIZED IN MOCK MODE: Real detection is disabled! "
                "Simulated synthetic detections will be emitted."
            )
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
                    cls_name.lower() in CLASS_ALIASES
                    for cls_name in self.model.names.values()
                )
                logger.info("YOLO Model loaded successfully. Road classes present: %s", self.has_road_classes)
            except Exception as exc:
                logger.error("Failed to load YOLO model: %s. Continuing with heuristic-only detection.", exc)
                self.model = None
                self.use_mock = False
                self.has_road_classes = False

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

    def _cv_defect_heuristic(
        self,
        frame: np.ndarray,
        vehicle_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Intelligent computer vision texture, contour, and geometric defect analyzer.

        Applies perspective trapezoid ROI masking, illumination normalization via residual
        estimation, adaptive lane marking suppression, vehicle exclusion, and multi-scale
        ridge and contour analysis to eliminate false positives on pristine highways.
        """
        detections: List[Dict[str, Any]] = []
        try:
            orig_h, orig_w = frame.shape[:2]

            # 1. Resize frame to standard WORK_WIDTH (640) for scale-invariant analysis
            work_w = settings.WORK_WIDTH
            scale = float(work_w) / float(orig_w)
            work_h = int(round(orig_h * scale))

            work_frame = cv2.resize(
                frame,
                (work_w, work_h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )

            # Resize vehicle mask if provided
            if vehicle_mask is not None:
                work_v_mask = cv2.resize(vehicle_mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
            else:
                work_v_mask = np.zeros((work_h, work_w), dtype=np.uint8)

            # 2. Perspective Trapezoid Polygon ROI Mask
            trapezoid_pts = np.array(
                [[int(fx * work_w), int(fy * work_h)] for fx, fy in settings.ROI_TRAPEZOID_FRACTIONS],
                dtype=np.int32,
            )
            roi_mask = np.zeros((work_h, work_w), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [trapezoid_pts], 255)
            roi_area = float(cv2.countNonZero(roi_mask))
            if roi_area <= 100.0:
                return detections

            gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

            # 3. Paint Masking: Standard and Adaptive Markings
            white_mask = cv2.inRange(gray, 180, 255)
            yellow_mask = cv2.inRange(hsv, np.array([15, 60, 120]), np.array([35, 255, 255]))
            raw_paint = cv2.bitwise_or(white_mask, yellow_mask)

            # Adaptive paint rule: (gray > road_median + 45) with low saturation
            prelim_road_pix = gray[(roi_mask > 0) & (work_v_mask == 0)]
            road_med_init = float(np.median(prelim_road_pix)) if len(prelim_road_pix) > 0 else 100.0
            adaptive_paint = ((gray > (road_med_init + 45.0)) & (hsv[:, :, 1] < 45)).astype(np.uint8) * 255
            raw_paint = cv2.bitwise_or(raw_paint, adaptive_paint)

            # Isolate lane markings and painted road markings (lines/stripes have stroke <= 25px or high aspect)
            lane_mask = np.zeros_like(gray)
            contours, _ = cv2.findContours(raw_paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 30:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(max(bw, bh)) / max(1, min(bw, bh))
                    if min(bw, bh) <= 25 or aspect >= 2.5:
                        cv2.drawContours(lane_mask, [cnt], -1, 255, thickness=-1)

            # Dilate lane mask about 15px to protect paint boundaries from edge detectors
            lane_mask_dilated = cv2.dilate(lane_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

            # 4. Illumination Normalization (Large-Kernel Background Residual)
            # Estimate unoccluded road illumination by closing small dark cavities and blurring
            close_bg = cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
            )
            x_r, y_r, w_r, h_r = cv2.boundingRect(trapezoid_pts)
            k = int(w_r / 8.0)
            if k % 2 == 0:
                k += 1
            k = max(31, k)
            background = cv2.GaussianBlur(close_bg, (k, k), 0)
            residual = gray.astype(np.float32) - background.astype(np.float32)

            # Valid clean asphalt pixels (ensure uint8 mask for OpenCV bitwise operations)
            valid_mask = ((roi_mask > 0) & (lane_mask_dilated == 0) & (work_v_mask == 0)).astype(np.uint8) * 255
            clean_road_residuals = residual[valid_mask > 0]
            if len(clean_road_residuals) < 200:
                return detections

            # Robust statistics: Median and MAD (Median Absolute Deviation)
            med_res = float(np.median(clean_road_residuals))
            mad = float(np.median(np.abs(clean_road_residuals - med_res)))
            sigma = float(max(1.0, 1.4826 * mad))

            road_gray_median = float(np.median(gray[valid_mask > 0]))
            road_sat_median = float(np.median(hsv[:, :, 1][valid_mask > 0]))

            # Sensitivity scaling
            sens = getattr(settings, "HEURISTIC_SENSITIVITY", "medium").lower()
            if sens == "low":
                s_fac = 1.15
            elif sens == "high":
                s_fac = 0.85
            else:
                s_fac = 1.0

            # Combined exclusion mask (lanes, vehicles, and ~12px ROI border margin)
            roi_border = cv2.bitwise_not(roi_mask)
            exclusion_base = cv2.bitwise_or(lane_mask_dilated, work_v_mask)
            exclusion_base = cv2.bitwise_or(exclusion_base, roi_border)
            exclusion_dilated = cv2.dilate(exclusion_base, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))

            dist_to_roi_border = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 3)

            pothole_candidates: List[Dict[str, Any]] = []
            crack_candidates: List[Dict[str, Any]] = []
            patch_candidates: List[Dict[str, Any]] = []

            # -------------------------------------------------------------
            # 5. POTHOLE DETECTION (Localized Dark Structural Cavities)
            # -------------------------------------------------------------
            pothole_thresh_val = -max(3.5 * sigma * s_fac, 28.0)
            pothole_bin = (residual < pothole_thresh_val).astype(np.uint8) * 255
            pothole_bin = cv2.bitwise_and(pothole_bin, cv2.bitwise_not(exclusion_dilated))
            pothole_opened = cv2.morphologyEx(
                pothole_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            pothole_cnts, _ = cv2.findContours(pothole_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in pothole_cnts:
                area = cv2.contourArea(cnt)
                # Reject small gravel noise and large shadow spans
                if area < 120 or area > (0.12 * roi_area):
                    continue

                # Reject blobs touching the ROI border
                min_border_dist = np.min([dist_to_roi_border[int(pt[0][1]), int(pt[0][0])] for pt in cnt])
                if min_border_dist <= 2.0:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh) if bh > 0 else 0
                if not (0.5 <= aspect <= 2.2):
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(1.0, hull_area)
                if solidity <= 0.80:
                    continue

                peri = cv2.arcLength(cnt, True)
                circularity = 4.0 * np.pi * area / max(1.0, peri * peri)
                if circularity <= 0.35:
                    continue

                # Surrounding ring contrast check (dilated by 15px minus blob)
                blob_m = np.zeros_like(gray)
                cv2.drawContours(blob_m, [cnt], -1, 255, -1)
                ring_m = cv2.dilate(blob_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
                ring_m = cv2.bitwise_and(ring_m, cv2.bitwise_not(blob_m))
                ring_m = cv2.bitwise_and(ring_m, valid_mask)

                if cv2.countNonZero(ring_m) >= 15:
                    blob_mean = float(np.mean(gray[blob_m > 0]))
                    ring_mean = float(np.mean(gray[ring_m > 0]))
                    contrast = ring_mean - blob_mean
                    if contrast < (25.0 * s_fac):
                        continue
                    if blob_mean > (road_gray_median - 20.0 * s_fac):
                        continue
                else:
                    continue

                # Evidence-based confidence
                conf = float(np.clip(0.40 + 0.35 * (contrast / 35.0) + 0.25 * solidity, 0.30, 0.95))
                if conf >= self.confidence_threshold:
                    pothole_candidates.append({
                        "class": "pothole",
                        "confidence": conf,
                        "bbox": (x, y, x + bw, y + bh),
                        "contour": cnt,
                    })

            # -------------------------------------------------------------
            # 6. CRACK DETECTION (Ridge Analysis via Black-Hat Filtering)
            # -------------------------------------------------------------
            kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
            crack_thresh_val = max(3.0 * sigma * s_fac, 18.0)
            _, crack_bin = cv2.threshold(blackhat, crack_thresh_val, 255, cv2.THRESH_BINARY)
            crack_bin = cv2.bitwise_and(crack_bin, cv2.bitwise_not(exclusion_dilated))

            crack_cnts, _ = cv2.findContours(crack_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in crack_cnts:
                peri = cv2.arcLength(cnt, False)
                est_length = peri / 2.0
                if est_length < 60.0:
                    continue

                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                elongation = float(max(rw, rh)) / max(1.0, min(rw, rh))
                if elongation < 4.0:
                    continue

                # Reject near-perfectly straight components in ANY orientation (line fit residual)
                if len(cnt) >= 5:
                    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx_f = float(vx[0]) if hasattr(vx, "__len__") else float(vx)
                    vy_f = float(vy[0]) if hasattr(vy, "__len__") else float(vy)
                    x0_f = float(x0[0]) if hasattr(x0, "__len__") else float(x0)
                    y0_f = float(y0[0]) if hasattr(y0, "__len__") else float(y0)
                    pts = cnt.reshape(-1, 2).astype(np.float32)
                    residuals = np.abs(-vy_f * (pts[:, 0] - x0_f) + vx_f * (pts[:, 1] - y0_f))
                    mean_residual = float(np.mean(residuals))
                    if mean_residual < 1.3:
                        continue  # Straight lane seam or painted stripe edge

                # Local contrast vs flanking pixels
                crack_m = np.zeros_like(gray)
                cv2.drawContours(crack_m, [cnt], -1, 255, -1)
                flank_m = cv2.dilate(crack_m, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
                flank_m = cv2.bitwise_and(flank_m, cv2.bitwise_not(crack_m))
                flank_m = cv2.bitwise_and(flank_m, valid_mask)

                if cv2.countNonZero(flank_m) >= 10:
                    crack_mean = float(np.mean(gray[crack_m > 0]))
                    flank_mean = float(np.mean(gray[flank_m > 0]))
                    contrast = flank_mean - crack_mean
                    if contrast < (15.0 * s_fac):
                        continue
                else:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                conf = float(np.clip(0.35 + 0.35 * (contrast / 25.0) + 0.25 * min(1.0, elongation / 8.0), 0.30, 0.95))
                if conf >= self.confidence_threshold:
                    crack_candidates.append({
                        "class": "crack",
                        "confidence": conf,
                        "bbox": (x, y, x + bw, y + bh),
                        "contour": cnt,
                    })

            # -------------------------------------------------------------
            # 7. ASPHALT PATCH & UTILITY DIG DETECTION
            # -------------------------------------------------------------
            patch_delta = max(24.0 * s_fac, 2.0 * sigma)
            patch_bin = (np.abs(gray.astype(np.float32) - road_gray_median) > patch_delta).astype(np.uint8) * 255
            patch_opened = cv2.morphologyEx(
                patch_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            )
            edges = cv2.Canny(gray, 40, 120)
            edges_dil = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            patch_cnts, _ = cv2.findContours(patch_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in patch_cnts:
                area = cv2.contourArea(cnt)
                # Keep patches bounded relative to working ROI
                if area < 800 or area > (0.15 * roi_area):
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                # Reject camera sensor/frame border artifacts
                if x <= 2 or y <= 2 or (x + bw) >= (work_w - 2) or (y + bh) >= (work_h - 2):
                    continue

                # Ensure centroid and majority of patch is within the perspective ROI
                c_x = int(x + bw / 2.0)
                c_y = int(y + bh / 2.0)
                if roi_mask[min(work_h - 1, c_y), min(work_w - 1, c_x)] == 0:
                    continue

                cnt_m = np.zeros_like(gray)
                cv2.drawContours(cnt_m, [cnt], -1, 255, -1)
                in_roi_pix = cv2.countNonZero(cv2.bitwise_and(cnt_m, roi_mask))
                if (float(in_roi_pix) / max(1.0, float(area))) < 0.70:
                    continue

                aspect = float(bw) / float(bh) if bh > 0 else 0
                if not (0.4 <= aspect <= 2.5):
                    continue

                rect_area = float(bw * bh)
                extent = float(area) / max(1.0, rect_area)
                if extent <= 0.55:
                    continue

                # Color family check (saturation difference)
                patch_sat = float(np.median(hsv[:, :, 1][cnt_m > 0]))
                if abs(patch_sat - road_sat_median) > 35.0:
                    continue

                # No overlap with actual lane markings or vehicles
                overlap_m = cv2.bitwise_and(cnt_m, cv2.bitwise_or(lane_mask, work_v_mask))
                if cv2.countNonZero(overlap_m) > 0:
                    continue

                # Visible edge ring along >= 60% of boundary
                b_mask = np.zeros_like(gray)
                cv2.drawContours(b_mask, [cnt], -1, 255, 1)
                b_total = cv2.countNonZero(b_mask)
                edge_hits = cv2.bitwise_and(edges_dil, b_mask)
                edge_ratio = float(cv2.countNonZero(edge_hits)) / max(1.0, float(b_total))
                if edge_ratio < 0.60:
                    continue

                patch_gray_med = float(np.median(gray[cnt_m > 0]))
                contrast = abs(patch_gray_med - road_gray_median)
                if contrast < patch_delta:
                    continue

                # Stricter condition for dig vs patch
                if extent >= 0.78 and contrast >= 28.0 and area >= 2000:
                    cls_type = "dig"
                else:
                    cls_type = "patch"

                conf = float(np.clip(0.40 + 0.35 * edge_ratio + 0.25 * extent, 0.30, 0.95))
                if conf >= self.confidence_threshold:
                    patch_candidates.append({
                        "class": cls_type,
                        "confidence": conf,
                        "bbox": (x, y, x + bw, y + bh),
                        "contour": cnt,
                    })

            # -------------------------------------------------------------
            # 8. CROSS-CLASS SUPPRESSION
            # -------------------------------------------------------------
            filtered_cracks = []
            for crack in crack_candidates:
                cb = crack["bbox"]
                suppressed = False
                for pot in pothole_candidates:
                    pb = pot["bbox"]
                    xi1, yi1 = max(cb[0], pb[0]), max(cb[1], pb[1])
                    xi2, yi2 = min(cb[2], pb[2]), min(cb[3], pb[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    p_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    union = c_area + p_area - inter
                    iou = float(inter) / float(union) if union > 0 else 0.0

                    c_center_x = (cb[0] + cb[2]) / 2.0
                    c_center_y = (cb[1] + cb[3]) / 2.0
                    is_contained = (pb[0] <= c_center_x <= pb[2] and pb[1] <= c_center_y <= pb[3])

                    if iou > 0.30 or is_contained:
                        suppressed = True
                        break
                if not suppressed:
                    filtered_cracks.append(crack)

            filtered_patches = []
            for patch in patch_candidates:
                ptb = patch["bbox"]
                suppressed = False
                for pot in pothole_candidates:
                    pb = pot["bbox"]
                    xi1, yi1 = max(ptb[0], pb[0]), max(ptb[1], pb[1])
                    xi2, yi2 = min(ptb[2], pb[2]), min(ptb[3], pb[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    pt_area = (ptb[2] - ptb[0]) * (ptb[3] - ptb[1])
                    p_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    union = pt_area + p_area - inter
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    if iou > 0.20:
                        suppressed = True
                        break
                if not suppressed:
                    filtered_patches.append(patch)

            # -------------------------------------------------------------
            # 9. FRAME GATE (Exposure, Glare, or Heavy Shadow Suppression)
            # -------------------------------------------------------------
            all_candidates = pothole_candidates + filtered_cracks + filtered_patches
            total_cand_area = sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in all_candidates)
            if total_cand_area > (0.35 * roi_area):
                logger.debug("Frame candidate gate triggered (coverage exceeds 35%% of ROI). Suppressing.")
                return []

            # -------------------------------------------------------------
            # 10. RESCALE COORDINATES TO ORIGINAL IMAGE DIMENSIONS
            # -------------------------------------------------------------
            for cand in all_candidates:
                bx1, by1, bx2, by2 = cand["bbox"]
                orig_x1 = int(round(bx1 / scale))
                orig_y1 = int(round(by1 / scale))
                orig_x2 = int(round(bx2 / scale))
                orig_y2 = int(round(by2 / scale))

                orig_x1 = max(0, min(orig_w, orig_x1))
                orig_y1 = max(0, min(orig_h, orig_y1))
                orig_x2 = max(0, min(orig_w, orig_x2))
                orig_y2 = max(0, min(orig_h, orig_y2))

                detections.append({
                    "class": cand["class"],
                    "confidence": cand["confidence"],
                    "bbox": (orig_x1, orig_y1, orig_x2, orig_y2),
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
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)

        if self.use_mock:
            raw_detections = self._simulate_frame_detections(frame_idx, w, h)
        else:
            if self.model is not None:
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
                            raw_cls_name = self.model.names.get(cls_id, "").lower()

                            # Vehicle and person obstacle masking
                            if raw_cls_name in COCO_OBSTACLE_CLASSES:
                                cv2.rectangle(vehicle_mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

                            # Normalize road defect classes via alias mapping
                            canonical_name = CLASS_ALIASES.get(raw_cls_name)
                            if canonical_name in TARGET_DEFECT_CLASSES:
                                raw_detections.append({
                                    "class": canonical_name,
                                    "confidence": conf,
                                    "bbox": (x1, y1, x2, y2),
                                })
                except Exception as exc:
                    logger.warning("YOLO inference failed: %s", exc)

            # Dilate vehicle mask by 15px to protect shadow edges
            if cv2.countNonZero(vehicle_mask) > 0:
                vehicle_mask = cv2.dilate(vehicle_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

            # If no custom road classes are in YOLO, run computer vision heuristic
            if not self.has_road_classes:
                raw_detections.extend(self._cv_defect_heuristic(frame, vehicle_mask=vehicle_mask))

        # Spatial tracking and deduplication with one-to-one matching and MIN_CONFIRM_HITS
        new_counts: Dict[str, int] = {"digs": 0, "patches": 0, "cracks": 0, "potholes": 0}

        detections_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for det in raw_detections:
            detections_by_class.setdefault(det["class"], []).append(det)

        tracks_by_class: Dict[str, List[DefectTrack]] = {}
        for trk in active_tracks:
            tracks_by_class.setdefault(trk.class_name, []).append(trk)

        updated_tracks: List[DefectTrack] = []

        for cls_name in TARGET_DEFECT_CLASSES:
            cls_dets = detections_by_class.get(cls_name, [])
            cls_trks = tracks_by_class.get(cls_name, [])

            if not cls_dets and not cls_trks:
                continue

            # Greedy one-to-one matching by best IoU
            pairs: List[Tuple[float, int, int]] = []
            for d_idx, det in enumerate(cls_dets):
                for t_idx, trk in enumerate(cls_trks):
                    if trk.is_match(det["bbox"], settings.TRACKER_IOU):
                        iou_val = trk.compute_iou(det["bbox"])
                        pairs.append((iou_val, d_idx, t_idx))

            pairs.sort(key=lambda p: p[0], reverse=True)

            matched_det_indices: Set[int] = set()
            matched_trk_indices: Set[int] = set()

            for _, d_idx, t_idx in pairs:
                if d_idx in matched_det_indices or t_idx in matched_trk_indices:
                    continue
                matched_det_indices.add(d_idx)
                matched_trk_indices.add(t_idx)

                trk = cls_trks[t_idx]
                det = cls_dets[d_idx]
                trk.bbox = det["bbox"]
                trk.last_frame = frame_idx
                trk.hits += 1
                updated_tracks.append(trk)

                # Count defect only once when track achieves MIN_CONFIRM_HITS
                if trk.hits >= settings.MIN_CONFIRM_HITS and not trk.counted:
                    trk.counted = True
                    key = f"{cls_name}s" if not cls_name.endswith("s") else cls_name
                    if key in new_counts:
                        new_counts[key] += 1

            # Unmatched detections spawn new tentative tracks
            for d_idx, det in enumerate(cls_dets):
                if d_idx not in matched_det_indices:
                    self._track_counter += 1
                    new_track = DefectTrack(self._track_counter, cls_name, det["bbox"], frame_idx)
                    new_track.hits = 1
                    new_track.counted = False
                    if new_track.hits >= settings.MIN_CONFIRM_HITS and not new_track.counted:
                        new_track.counted = True
                        key = f"{cls_name}s" if not cls_name.endswith("s") else cls_name
                        if key in new_counts:
                            new_counts[key] += 1
                    updated_tracks.append(new_track)

        # Retain un-updated active tracks that haven't expired
        track_dict = {t.track_id: t for t in active_tracks}
        for t in updated_tracks:
            track_dict[t.track_id] = t

        max_age_frames = settings.TRACK_MAX_AGE * self.frame_skip
        surviving_tracks = [
            t for t in track_dict.values()
            if (frame_idx - t.last_frame) <= max_age_frames
        ]

        # Draw bounding boxes ONLY for confirmed tracks
        for trk in surviving_tracks:
            if trk.hits >= settings.MIN_CONFIRM_HITS:
                color = settings.DEFECT_COLORS.get(trk.class_name, (0, 255, 0))
                bbox = trk.bbox
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"#{trk.track_id} {trk.class_name.upper()} (hits:{trk.hits})"
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
                    # NOTE: Do NOT clear active_tracks at segment boundaries!
                    # Defects spanning segments are carried over with counted flag to prevent duplicate counts.

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
