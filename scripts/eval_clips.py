"""Evaluation harness to benchmark defect detection and PCI across clean and damaged test clips.

Runs process_video on every clip in data/test_clean/ and data/test_damaged/,
reporting per-segment defects, segment PCI, and overall PCI.
Outputs metrics to baseline.json or a specified JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List
import cv2
import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logger import get_logger
from config.settings import settings
from vision.detector import RoadDefectDetector

logger = get_logger("eval_clips")


def generate_clean_clip(output_path: Path, total_frames: int = 200, with_shadows: bool = False) -> None:
    """Generates a clean synthetic road clip with zero physical defects."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))

    np.random.seed(101 if with_shadows else 42)

    for frame_idx in range(total_frames):
        # Base asphalt background
        frame = np.full((h, w, 3), 92, dtype=np.uint8)
        # Add realistic micro-texture noise
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # White boundary lane lines
        cv2.line(frame, (90, 0), (90, h), (255, 255, 255), 4)
        cv2.line(frame, (w - 90, 0), (w - 90, h), (255, 255, 255), 4)

        # Moving dashed centerline
        dash_offset = (frame_idx * 14) % 80
        for y in range(-80 + dash_offset, h + 80, 80):
            cv2.line(frame, (w // 2, y), (w // 2, y + 45), (0, 220, 255), 4)

        # Add moving tree/ambient shadow bands if with_shadows
        if with_shadows:
            shadow_y = int((frame_idx * 7) % (h + 100)) - 50
            overlay = frame.copy()
            pts = np.array([[120, shadow_y], [w - 120, shadow_y + 40], [w - 120, shadow_y + 90], [120, shadow_y + 50]], np.int32)
            cv2.fillPoly(overlay, [pts], (45, 45, 45))
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        writer.write(frame)

    writer.release()
    logger.info("Generated clean clip at: %s", output_path)


def ensure_test_clips(clean_dir: Path, damaged_dir: Path) -> None:
    """Ensures test clips exist in data/test_clean/ and data/test_damaged/."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    damaged_dir.mkdir(parents=True, exist_ok=True)

    # Check clean clips
    clean_clips = list(clean_dir.glob("*.mp4")) + list(clean_dir.glob("*.avi"))
    if not clean_clips:
        logger.info("No clean clips found in %s. Synthesizing test_clean clips...", clean_dir)
        generate_clean_clip(clean_dir / "clean_highway.mp4", total_frames=200, with_shadows=False)
        generate_clean_clip(clean_dir / "clean_shadows.mp4", total_frames=200, with_shadows=True)

    # Check damaged clips
    damaged_clips = list(damaged_dir.glob("*.mp4")) + list(damaged_dir.glob("*.avi"))
    if not damaged_clips:
        logger.info("No damaged clips found in %s. Sourcing damaged clip...", damaged_dir)
        sample_path = PROJECT_ROOT / "data" / "sample_videos" / "sample_road.mp4"
        if not sample_path.exists():
            sample_path = PROJECT_ROOT / "data" / "videos" / "sample_road.mp4"
        if sample_path.exists():
            target = damaged_dir / "damaged_road.mp4"
            shutil.copy2(str(sample_path), str(target))
            logger.info("Copied %s -> %s", sample_path, target)
        else:
            # Generate sample damaged video using generate_sample_video.py
            from data.generate_sample_video import generate_sample_road_video
            target = damaged_dir / "damaged_road.mp4"
            generate_sample_road_video(str(target), total_frames=200)
            logger.info("Synthesized damaged road clip at %s", target)


def evaluate_clips(clean_dir: Path, damaged_dir: Path) -> Dict[str, Any]:
    """Runs process_video across clean and damaged test clips and aggregates results."""
    detector = RoadDefectDetector()

    results: Dict[str, Any] = {
        "clean_clips": {},
        "damaged_clips": {},
        "summary": {
            "total_clean_clips": 0,
            "total_damaged_clips": 0,
            "clean_false_positive_potholes": 0,
            "clean_false_positive_cracks": 0,
        },
    }

    print("\n" + "=" * 80)
    print("ROAD DEFECT DETECTOR EVALUATION HARNESS")
    print("=" * 80)

    # 1. Clean Clips
    clean_files = sorted(list(clean_dir.glob("*.mp4")) + list(clean_dir.glob("*.avi")))
    print(f"\n[Evaluating Clean Clips] ({len(clean_files)} clips found in {clean_dir})")
    print("-" * 80)

    for cfile in clean_files:
        report = detector.process_video(str(cfile), display=False)
        clip_data = {
            "filename": cfile.name,
            "total_segments": report.get("total_segments_analyzed", len(report.get("segment_details", []))),
            "overall_pci": report["summary"]["overall_pci"],
            "total_defects": report["summary"]["total_defects"],
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "frame_start": s["frame_start"],
                    "frame_end": s["frame_end"],
                    "defects": s["defects"],
                    "pci_score": s["pci_score"],
                }
                for s in report.get("segment_details", [])
            ],
        }
        results["clean_clips"][cfile.name] = clip_data
        results["summary"]["total_clean_clips"] += 1
        results["summary"]["clean_false_positive_potholes"] += clip_data["total_defects"].get("potholes", 0)
        results["summary"]["clean_false_positive_cracks"] += clip_data["total_defects"].get("cracks", 0)

        print(f"Clip: {cfile.name} | Overall PCI: {clip_data['overall_pci']:.1f}% | Total Defects: {clip_data['total_defects']}")
        for seg in clip_data["segments"]:
            print(f"  -> Seg #{seg['segment_id']} (Frames {seg['frame_start']}-{seg['frame_end']}): PCI={seg['pci_score']:.1f}% | Defects={seg['defects']}")

    # 2. Damaged Clips
    damaged_files = sorted(list(damaged_dir.glob("*.mp4")) + list(damaged_dir.glob("*.avi")))
    print(f"\n[Evaluating Damaged Clips] ({len(damaged_files)} clips found in {damaged_dir})")
    print("-" * 80)

    for dfile in damaged_files:
        report = detector.process_video(str(dfile), display=False)
        clip_data = {
            "filename": dfile.name,
            "total_segments": report.get("total_segments_analyzed", len(report.get("segment_details", []))),
            "overall_pci": report["summary"]["overall_pci"],
            "total_defects": report["summary"]["total_defects"],
            "segments": [
                {
                    "segment_id": s["segment_id"],
                    "frame_start": s["frame_start"],
                    "frame_end": s["frame_end"],
                    "defects": s["defects"],
                    "pci_score": s["pci_score"],
                }
                for s in report.get("segment_details", [])
            ],
        }
        results["damaged_clips"][dfile.name] = clip_data
        results["summary"]["total_damaged_clips"] += 1

        print(f"Clip: {dfile.name} | Overall PCI: {clip_data['overall_pci']:.1f}% | Total Defects: {clip_data['total_defects']}")
        for seg in clip_data["segments"]:
            print(f"  -> Seg #{seg['segment_id']} (Frames {seg['frame_start']}-{seg['frame_end']}): PCI={seg['pci_score']:.1f}% | Defects={seg['defects']}")

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {len(clean_files)} Clean Clips, {len(damaged_files)} Damaged Clips")
    print("=" * 80 + "\n")

    return results


def compare_with_baseline(current: Dict[str, Any], baseline_file: Path) -> None:
    """Compares current evaluation results with baseline.json."""
    if not baseline_file.exists():
        print(f"Baseline file '{baseline_file}' not found. Cannot perform comparison.")
        return

    with open(baseline_file, "r") as f:
        baseline = json.load(f)

    print("\n" + "=" * 80)
    print("COMPARISON: CURRENT vs BASELINE")
    print("=" * 80)

    # Clean clips comparison
    print("\n--- CLEAN CLIPS ---")
    for name, cdata in current.get("clean_clips", {}).items():
        bdata = baseline.get("clean_clips", {}).get(name, {})
        b_pci = bdata.get("overall_pci", 0)
        c_pci = cdata.get("overall_pci", 0)
        b_def = bdata.get("total_defects", {})
        c_def = cdata.get("total_defects", {})

        print(f"Clip '{name}':")
        print(f"  PCI:       {b_pci:.1f}% -> {c_pci:.1f}% ({c_pci - b_pci:+.1f}%)")
        print(f"  Potholes:  {b_def.get('potholes', 0)} -> {c_def.get('potholes', 0)}")
        print(f"  Cracks:    {b_def.get('cracks', 0)} -> {c_def.get('cracks', 0)}")
        print(f"  Patches:   {b_def.get('patches', 0)} -> {c_def.get('patches', 0)}")
        print(f"  Digs:      {b_def.get('digs', 0)} -> {c_def.get('digs', 0)}")

        # Verification check
        potholes = c_def.get('potholes', 0)
        cracks_per_seg = [s['defects'].get('cracks', 0) for s in cdata.get('segments', [])]
        max_cracks = max(cracks_per_seg) if cracks_per_seg else 0
        pci_ok = c_pci >= 90.0
        potholes_ok = potholes == 0
        cracks_ok = max_cracks <= 1

        print(f"  ACCEPTANCE: Potholes==0: {'✅' if potholes_ok else '❌'} | Cracks/seg<=1: {'✅' if cracks_ok else '❌'} | PCI>=90: {'✅' if pci_ok else '❌'}")

    # Damaged clips comparison
    print("\n--- DAMAGED CLIPS ---")
    for name, ddata in current.get("damaged_clips", {}).items():
        bdata = baseline.get("damaged_clips", {}).get(name, {})
        b_pci = bdata.get("overall_pci", 0)
        d_pci = ddata.get("overall_pci", 0)
        b_def = bdata.get("total_defects", {})
        d_def = ddata.get("total_defects", {})

        b_total = sum(b_def.values())
        d_total = sum(d_def.values())
        retention = (d_total / max(1, b_total)) * 100.0

        print(f"Clip '{name}':")
        print(f"  PCI:       {b_pci:.1f}% -> {d_pci:.1f}%")
        print(f"  Defects:   {b_def} -> {d_def}")
        print(f"  Total:     {b_total} -> {d_total} ({retention:.1f}% retention vs baseline)")
        print(f"  ACCEPTANCE: Retention >= 80%: {'✅' if retention >= 80.0 else '⚠️ (Check drop)'}")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate defect detection across clean and damaged test clips.")
    parser.add_argument("--clean-dir", default=str(PROJECT_ROOT / "data" / "test_clean"), help="Path to clean clips directory")
    parser.add_argument("--damaged-dir", default=str(PROJECT_ROOT / "data" / "test_damaged"), help="Path to damaged clips directory")
    parser.add_argument("--output", "-o", default=None, help="Destination JSON file (defaults to baseline.json or eval_results.json if comparing)")
    parser.add_argument("--compare", "-c", default=None, help="Baseline JSON file to compare with")

    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    damaged_dir = Path(args.damaged_dir)

    ensure_test_clips(clean_dir, damaged_dir)
    results = evaluate_clips(clean_dir, damaged_dir)

    out_file = args.output
    if not out_file:
        out_file = str(PROJECT_ROOT / ("eval_results.json" if args.compare else "baseline.json"))

    out_path = Path(out_file)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to: {out_path.resolve()}")

    if args.compare:
        compare_with_baseline(results, Path(args.compare))


if __name__ == "__main__":
    main()
