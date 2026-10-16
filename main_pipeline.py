"""Main Orchestration Pipeline: OpenCV-YOLO Defect Detection to RL Maintenance Decision.

Integrates video ingestion, YOLO object detection, PCI calculation, and
Deep Q-Network (DQN) policy inference into an end-to-end processing pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from stable_baselines3 import DQN

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RL_ACTION_NAMES,
    RL_MODEL_PATH,
    SAMPLE_VIDEOS_DIR,
    setup_logging,
)
from vision.detector import RoadDefectDetector
from vision.pci_engine import build_rl_state, calculate_pci, get_pci_condition_category

logger = setup_logging(__name__)


def run_pipeline(
    video_path: str,
    rl_model_path: Optional[str] = None,
    road_age_years: float = 3.0,
    display: bool = False,
    output_video_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the end-to-end road inspection and maintenance recommendation pipeline.

    Args:
        video_path: Path to target input video (.mp4 or .avi).
        rl_model_path: Path to trained DQN model zip file.
        road_age_years: Estimated physical age of the road segment in years.
        display: If True, renders OpenCV live detection window.
        output_video_path: Optional path to save annotated output video.
        output_json_path: Optional path to save JSON inspection report.

    Returns:
        Dict[str, Any]: Complete analysis report including defect counts, PCI, and RL predictions.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Input video file not found at: {video_path}")

    target_rl_model = Path(rl_model_path if rl_model_path else RL_MODEL_PATH)
    if not target_rl_model.exists():
        raise FileNotFoundError(
            f"Trained RL model not found at '{target_rl_model}'. "
            "Please train the model first using 'python -m rl_engine.train'."
        )

    logger.info("Initializing Video Defect Detector...")
    detector = RoadDefectDetector()

    logger.info("Loading RL Maintenance Policy from: %s", target_rl_model)
    rl_model = DQN.load(str(target_rl_model))

    logger.info("Executing OpenCV & YOLO detection on: %s", video_file.name)
    detection_results = detector.process_video(
        video_path=str(video_file),
        display=display,
        output_path=output_video_path,
    )

    enriched_segments: List[Dict[str, Any]] = []
    recommended_actions: List[str] = []

    print("\n" + "=" * 65)
    print("      AI SMART ROAD DEFECT DETECTION & PREDICTIVE MAINTENANCE")
    print("=" * 65)
    print(f"Video File: {video_file.name}")
    print(f"Total Frames: {detection_results['total_frames']} | Analyzed Segments: {len(detection_results['segment_details'])}")
    print("=" * 65)

    for seg in detection_results["segment_details"]:
        seg_id = seg["segment_id"]
        defects = seg["defects"]
        pci = seg["pci_score"]
        condition = get_pci_condition_category(pci)

        # Build strict 5-element state vector: [PCI, Digs, Patches, Cracks, Age]
        state_vector = build_rl_state(pci=pci, defects_dict=defects, road_age_years=road_age_years)

        # Run RL inference
        action_idx, _ = rl_model.predict(state_vector, deterministic=True)
        action_int = int(action_idx)
        action_name = RL_ACTION_NAMES.get(action_int, "Do Nothing")

        # Map to descriptive action guidance
        if action_int == 0:
            guidance = "Optimal Condition - No Immediate Action Required"
        elif action_int == 1:
            guidance = "Preventive Repair Required (30-day maintenance window)"
        else:
            guidance = "Major Resurfacing Required (Immediate reconstruction priority)"

        recommended_actions.append(action_name)

        enriched_segment = {
            "segment_id": seg_id,
            "frame_range": f"{seg['frame_start']} - {seg['frame_end']}",
            "pci_score": pci,
            "condition": condition,
            "defects": defects,
            "rl_state_vector": state_vector.tolist(),
            "rl_recommended_action": action_int,
            "action_name": action_name,
            "action_guidance": guidance,
        }
        enriched_segments.append(enriched_segment)

        # Print formatted summary report
        print(f"Segment ID: {seg_id}")
        print(
            f"Defects Found: {defects['digs']} Digs, {defects['patches']} Patches, "
            f"{defects['cracks']} Cracks, {defects['potholes']} Potholes"
        )
        print(f"Computed PCI: {pci:.1f} / 100.0 (Status: {condition})")
        print(f"Recommended RL Action: {guidance}")
        print("-" * 65)

    # Compute overall summary
    total_defects = detection_results["summary"]["total_defects"]
    overall_pci = calculate_pci(total_defects)
    overall_condition = get_pci_condition_category(overall_pci)

    # Overall recommendation based on most urgent segment or overall PCI
    if any(s["rl_recommended_action"] == 2 for s in enriched_segments) or overall_pci < 40.0:
        overall_recommendation = "Major Resurfacing Required"
    elif any(s["rl_recommended_action"] == 1 for s in enriched_segments) or overall_pci < 70.0:
        overall_recommendation = "Preventive Repair Required"
    else:
        overall_recommendation = "Do Nothing (Good Condition)"

    print("\nOVERALL INSPECTION SUMMARY:")
    print(f"Total Defects: {total_defects}")
    print(f"Overall PCI: {overall_pci:.1f} / 100.0 ({overall_condition})")
    print(f"Final Fleet Recommendation: {overall_recommendation}")
    print("=" * 65 + "\n")

    full_report: Dict[str, Any] = {
        "status": "success",
        "filename": video_file.name,
        "total_segments_analyzed": len(enriched_segments),
        "summary": {
            "overall_pci": overall_pci,
            "overall_condition": overall_condition,
            "recommended_action": overall_recommendation,
            "total_defects": total_defects,
        },
        "segment_details": enriched_segments,
    }

    if output_json_path:
        out_json = Path(output_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        logger.info("Inspection report exported to: %s", out_json)

    return full_report


def main() -> None:
    """CLI Entry point for the main orchestration pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-End AI Road Defect Detection and RL Maintenance Pipeline."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(SAMPLE_VIDEOS_DIR / "sample_road.mp4"),
        help="Path to input video file (default: data/sample_videos/sample_road.mp4).",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=RL_MODEL_PATH,
        help=f"Path to trained RL model (default: {RL_MODEL_PATH}).",
    )
    parser.add_argument(
        "--road-age",
        type=float,
        default=3.0,
        help="Estimated age of the road in years (default: 3.0).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display OpenCV detection window during processing.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to export annotated video.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/outputs/inspection_report.json",
        help="Optional path to export JSON inspection report.",
    )

    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        rl_model_path=args.rl_model,
        road_age_years=args.road_age,
        display=args.display,
        output_video_path=args.output_video,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
