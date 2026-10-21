"""Pipeline for Extracting, Training, and Testing AI Model on Real Road Videos.

This script processes the 3 real inspection videos located in the `videos/` directory:
1. 4695859-uhd_3840_2160_30fps.mp4 (Scenic Paved Highway 4K)
2. 16433509_2160_3840_60fps.mp4 (Mountain Pass Trail 4K 60fps)
3. 20258332-hd_1080_1920_30fps.mp4 (Urban Arterial Dashcam 1080p)

Workflow:
1. Video Defect Extraction: Uses OpenCV & RoadDefectDetector to extract real defect
   counts (digs, patches, cracks, potholes) and ASTM D6433 PCI scores for each video segment.
2. Empirical Dataset Construction: Assembles real-world observation states.
3. RL Policy Training: Trains a Deep Q-Network (DQN) agent on the empirical distribution
   in a specialized Gymnasium environment.
4. Comprehensive Model Testing: Runs policy inference on every segment of all 3 videos,
   evaluating maintenance recommendations, budget allocation, and pavement lifecycle outcomes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ACTION_DO_NOTHING,
    ACTION_MAJOR_RESURFACING,
    ACTION_PREVENTIVE,
    MAX_PCI,
    MIN_PCI,
    RL_ACTION_COSTS,
    RL_ACTION_NAMES,
    RL_MODEL_PATH,
    setup_logging,
)
from rl_engine.road_env import RoadMaintenanceEnv
from vision.detector import RoadDefectDetector
from vision.pci_engine import build_rl_state, calculate_pci, get_pci_condition_category

logger = setup_logging(__name__)

REAL_VIDEOS = [
    "4695859-uhd_3840_2160_30fps.mp4",
    "16433509_2160_3840_60fps.mp4",
    "20258332-hd_1080_1920_30fps.mp4",
]

DATASET_CACHE_PATH = PROJECT_ROOT / "data" / "real_video_dataset.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "eval_real_videos.json"


class EmpiricalRealVideoRoadEnv(RoadMaintenanceEnv):
    """Gymnasium environment initialized with real-world road defect distributions.

    Samples initial states directly from empirical road segments observed across
    the real video dataset, exposing the DQN policy to realistic pavement defect
    correlations, crack densities, and PCI deterioration profiles.
    """

    def __init__(
        self,
        empirical_states: List[Dict[str, Any]],
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.empirical_states = empirical_states
        if not self.empirical_states:
            raise ValueError("Empirical states list cannot be empty for EmpiricalRealVideoRoadEnv")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options and not options.get("randomize", False):
            # Deterministic reset via explicit options
            self.pci = float(np.clip(options.get("pci", 100.0), MIN_PCI, MAX_PCI))
            self.digs = int(np.clip(options.get("digs", 0), 0, 50))
            self.patches = int(np.clip(options.get("patches", 0), 0, 50))
            self.cracks = int(np.clip(options.get("cracks", 0), 0, 50))
            self.road_age = float(np.clip(options.get("road_age", 0.0), 0.0, 30.0))
        else:
            # Sample directly from real video segment observations
            idx = int(self.np_random.integers(0, len(self.empirical_states)))
            sample = self.empirical_states[idx]

            base_pci = float(sample.get("pci_score", 100.0))
            defects = sample.get("defects", {})
            base_digs = int(defects.get("digs", 0))
            base_patches = int(defects.get("patches", 0))
            base_cracks = int(defects.get("cracks", 0))

            # Add slight realistic stochastic jitter for policy robustness
            pci_noise = float(self.np_random.uniform(-3.0, 3.0))
            self.pci = float(np.clip(base_pci + pci_noise, MIN_PCI, MAX_PCI))
            self.digs = int(np.clip(base_digs + self.np_random.integers(-1, 2), 0, 50))
            self.patches = int(np.clip(base_patches + self.np_random.integers(-1, 2), 0, 50))
            self.cracks = int(np.clip(base_cracks + self.np_random.integers(-1, 2), 0, 50))
            self.road_age = float(np.clip(self.np_random.uniform(1.0, 8.0), 0.0, 30.0))

        observation = self._get_obs()
        info: Dict[str, Any] = {
            "pci": self.pci,
            "defects": {
                "digs": self.digs,
                "patches": self.patches,
                "cracks": self.cracks,
            },
            "road_age": self.road_age,
        }
        return observation, info


def extract_real_video_dataset(
    video_filenames: List[str],
    force_reextract: bool = False,
) -> Dict[str, Any]:
    """Inspects real videos via RoadDefectDetector to construct empirical dataset."""
    if DATASET_CACHE_PATH.exists() and not force_reextract:
        logger.info("Found cached real video dataset at: %s", DATASET_CACHE_PATH)
        try:
            with open(DATASET_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify all requested videos are present
            cached_videos = {v["filename"] for v in data.get("videos", [])}
            if all(name in cached_videos for name in video_filenames):
                logger.info("All %d requested videos found in cache. Reusing dataset.", len(video_filenames))
                return data
            logger.info("Some requested videos missing from cache. Re-extracting...")
        except Exception as e:
            logger.warning("Error reading cached dataset (%s). Re-extracting...", e)

    logger.info("Initializing RoadDefectDetector for real video feature extraction...")
    detector = RoadDefectDetector(use_mock=False)

    videos_data = []
    all_empirical_segments = []

    for name in video_filenames:
        video_path = PROJECT_ROOT / "videos" / name
        if not video_path.exists():
            logger.error("Real video file not found at: %s", video_path)
            continue

        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logger.info(
            "Extracting defects from: %s (%dx%d, %.1ffps, %d frames)...",
            name,
            w,
            h,
            fps,
            total_frames,
        )

        t_start = time.perf_counter()
        results = detector.process_video(str(video_path))
        t_elapsed = round(time.perf_counter() - t_start, 2)

        segment_details = results.get("segment_details", [])
        logger.info(
            "Extracted %d segments from %s in %.2fs (Overall PCI: %.1f)",
            len(segment_details),
            name,
            t_elapsed,
            results.get("summary", {}).get("overall_pci", 100.0),
        )

        video_record = {
            "filename": name,
            "filepath": str(video_path),
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_frames,
            "extraction_time_sec": t_elapsed,
            "summary": results.get("summary", {}),
            "segment_details": segment_details,
        }
        videos_data.append(video_record)

        for seg in segment_details:
            pci = seg["pci_score"]
            cond = seg.get("condition") or get_pci_condition_category(pci)
            seg["condition"] = cond
            all_empirical_segments.append({
                "video_filename": name,
                "segment_id": seg["segment_id"],
                "pci_score": pci,
                "condition": cond,
                "defects": seg["defects"],
            })

    dataset = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_videos": len(videos_data),
        "total_empirical_segments": len(all_empirical_segments),
        "videos": videos_data,
        "empirical_segments": all_empirical_segments,
    }

    DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    logger.info("Dataset successfully serialized to: %s", DATASET_CACHE_PATH)

    return dataset


def train_on_real_videos(
    empirical_segments: List[Dict[str, Any]],
    timesteps: int = 30000,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> DQN:
    """Trains DQN policy in Gymnasium environment grounded in real video states."""
    target_path = Path(save_path if save_path else RL_MODEL_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("TRAINING DQN MAINTENANCE AGENT GROUNDED IN REAL VIDEOS")
    logger.info("=" * 65)
    logger.info("Empirical Segment Pool: %d samples", len(empirical_segments))
    logger.info("Target Timesteps: %d | Learning Rate: 0.001 | Seed: %d", timesteps, seed)

    raw_env = EmpiricalRealVideoRoadEnv(empirical_states=empirical_segments)
    env = Monitor(raw_env)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_time = round(time.perf_counter() - t0, 2)

    logger.info("Training complete in %.2f seconds.", train_time)
    model.save(str(target_path))
    logger.info("Primary policy saved to: %s", target_path)

    # Save dedicated named backup as well
    backup_path = target_path.parent / "dqn_road_policy_real_videos.zip"
    shutil.copyfile(str(target_path), str(backup_path))
    logger.info("Policy backup copy created at: %s", backup_path)

    return model


def test_model_on_real_videos(
    model_path: str,
    dataset: Dict[str, Any],
    road_age_years: float = 3.5,
) -> Dict[str, Any]:
    """Tests the trained policy on every segment of the 3 real videos."""
    logger.info("=" * 65)
    logger.info("TESTING POLICY ON ALL REAL VIDEO CORRIDORS")
    logger.info("=" * 65)

    target_path = Path(model_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found: {target_path}")

    model = DQN.load(str(target_path))

    test_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(target_path),
        "test_road_age_years": road_age_years,
        "video_evaluations": [],
    }

    action_names = {
        ACTION_DO_NOTHING: "Do Nothing (Optimal Condition)",
        ACTION_PREVENTIVE: "Preventive Repair Required (30-day window)",
        ACTION_MAJOR_RESURFACING: "Major Resurfacing Required (Immediate Action)",
    }

    for video in dataset.get("videos", []):
        filename = video["filename"]
        resolution = video.get("resolution", "N/A")
        segments = video.get("segment_details", [])

        action_counts = {0: 0, 1: 0, 2: 0}
        total_intervention_cost = 0.0
        segment_results = []

        print(f"\nEvaluating Corridor Video: {filename} ({resolution})")
        print(f"Total Segments: {len(segments)} | Estimated Pavement Age: {road_age_years:.1f} yrs")
        print("-" * 75)
        print(f"{'Seg ID':<8} | {'Frames':<12} | {'PCI':<6} | {'Status':<12} | {'Defects':<16} | {'Recommended RL Action'}")
        print("-" * 75)

        for seg in segments:
            seg_id = seg["segment_id"]
            pci = seg["pci_score"]
            condition = seg.get("condition") or get_pci_condition_category(pci)
            defects = seg["defects"]

            # Construct state vector
            state = build_rl_state(pci=pci, defects_dict=defects, road_age_years=road_age_years)
            act_idx, _ = model.predict(state, deterministic=True)
            act_int = int(act_idx)
            act_name = action_names.get(act_int, f"Action {act_int}")

            action_counts[act_int] += 1
            total_intervention_cost += RL_ACTION_COSTS.get(act_int, 0.0)

            defect_str = f"D:{defects['digs']} P:{defects['patches']} C:{defects['cracks']} H:{defects['potholes']}"
            frame_range = f"{seg['frame_start']}-{seg['frame_end']}"

            print(f"{seg_id:<8} | {frame_range:<12} | {pci:<6.1f} | {condition:<12} | {defect_str:<16} | {act_name}")

            segment_results.append({
                "segment_id": seg_id,
                "frame_start": seg["frame_start"],
                "frame_end": seg["frame_end"],
                "pci_score": pci,
                "condition": condition,
                "defects": defects,
                "recommended_action_code": act_int,
                "recommended_action_name": act_name,
                "action_cost": RL_ACTION_COSTS.get(act_int, 0.0),
            })

        avg_pci = float(np.mean([s["pci_score"] for s in segments])) if segments else 100.0
        summary_rec = (
            "Major Resurfacing Required" if action_counts[2] > 0
            else ("Preventive Repair Required" if action_counts[1] > 0 else "Do Nothing (Corridor in Good Health)")
        )

        print("-" * 75)
        print(f"CORRIDOR SUMMARY: Mean PCI = {avg_pci:.1f} ({get_pci_condition_category(avg_pci)})")
        print(
            f"Actions Distribution -> Optimal: {action_counts[0]}, "
            f"Preventive: {action_counts[1]}, Resurfacing: {action_counts[2]}"
        )
        print(f"Total Corridor Intervention Cost: ${total_intervention_cost:,.0f}k | Primary Decision: {summary_rec}")
        print("=" * 75)

        test_results["video_evaluations"].append({
            "filename": filename,
            "resolution": resolution,
            "total_segments": len(segments),
            "average_pci": round(avg_pci, 1),
            "overall_condition": get_pci_condition_category(avg_pci),
            "action_distribution": action_counts,
            "total_intervention_cost": total_intervention_cost,
            "primary_corridor_decision": summary_rec,
            "segment_evaluations": segment_results,
        })

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info("Evaluation results saved to: %s", EVAL_RESULTS_PATH)

    return test_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test AI models on 3 real road videos.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30000,
        help="Training timesteps for DQN agent (default: 30000)",
    )
    parser.add_argument(
        "--road-age",
        type=float,
        default=3.5,
        help="Estimated pavement surface age in years for test evaluation (default: 3.5)",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force re-extraction of video frames and defects even if cached",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract dataset without training or testing",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the current trained model on real videos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training (default: 42)",
    )

    args = parser.parse_args()

    # Step 1: Extract real video dataset
    dataset = extract_real_video_dataset(
        video_filenames=REAL_VIDEOS,
        force_reextract=args.force_reextract,
    )

    if args.extract_only:
        print(f"\n[OK] Extracted {dataset['total_empirical_segments']} empirical segments from {dataset['total_videos']} videos.")
        return

    # Step 2: Train DQN Agent on Real Video States (if not eval-only)
    if not args.eval_only:
        train_on_real_videos(
            empirical_segments=dataset["empirical_segments"],
            timesteps=args.timesteps,
            save_path=str(RL_MODEL_PATH),
            seed=args.seed,
        )

    # Step 3: Test Model on All Real Videos
    test_model_on_real_videos(
        model_path=str(RL_MODEL_PATH),
        dataset=dataset,
        road_age_years=args.road_age,
    )


if __name__ == "__main__":
    main()
