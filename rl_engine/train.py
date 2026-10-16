"""Reinforcement Learning Policy Training Script for Road Maintenance.

Trains a Deep Q-Network (DQN) agent on the custom Gymnasium RoadMaintenanceEnv
using Stable-Baselines3, saves the policy weights, and performs evaluation rollouts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODELS_DIR,
    RL_ACTION_NAMES,
    RL_MODEL_PATH,
    setup_logging,
)

logger = setup_logging(__name__)

from rl_engine.road_env import RoadMaintenanceEnv


def train_dqn_agent(
    total_timesteps: int = 50000,
    save_path: Optional[str] = None,
    learning_rate: float = 1e-3,
    buffer_size: int = 10000,
    seed: int = 42,
) -> DQN:
    """Instantiates the road environment and trains a DQN policy agent.

    Args:
        total_timesteps: Total environment interaction steps for training.
        save_path: Destination path for the trained policy zip file.
        learning_rate: Optimizer learning rate (default 1e-3).
        buffer_size: Experience replay buffer capacity.
        seed: Random seed for environment and policy reproducibility.

    Returns:
        DQN: The trained Stable-Baselines3 DQN model instance.
    """
    model_output_path = Path(save_path if save_path else RL_MODEL_PATH)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing RoadMaintenanceEnv for training (Seed: %d)...", seed)
    raw_env = RoadMaintenanceEnv()
    env = Monitor(raw_env)

    logger.info(
        "Creating DQN Agent (Policy: MlpPolicy, LR: %s, BufferSize: %d, Timesteps: %d)...",
        learning_rate,
        buffer_size,
        total_timesteps,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
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

    logger.info("Starting DQN policy training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    logger.info("Training complete. Exporting model weights to: %s", model_output_path)
    model.save(str(model_output_path))
    logger.info("Model saved successfully.")

    return model


def evaluate_policy(
    model_path: Optional[str] = None,
    num_episodes: int = 5,
    seed: int = 100,
) -> List[Dict[str, float]]:
    """Evaluates a trained maintenance policy over multiple test episodes.

    Args:
        model_path: Path to the serialized DQN model zip file.
        num_episodes: Number of complete test episodes to simulate.
        seed: Base random seed for test evaluation.

    Returns:
        List[Dict[str, float]]: Performance metrics per evaluation episode.
    """
    target_path = Path(model_path if model_path else RL_MODEL_PATH)
    if not target_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {target_path}")

    logger.info("Loading policy model from: %s", target_path)
    model = DQN.load(str(target_path))

    env = RoadMaintenanceEnv()
    episode_results: List[Dict[str, float]] = []

    logger.info("=" * 65)
    logger.info("BEGINNING POLICY EVALUATION (%d EPISODES)", num_episodes)
    logger.info("=" * 65)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + episode, options={"randomize": True})
        cumulative_reward: float = 0.0
        step_count: int = 0
        actions_taken: List[str] = []

        logger.info("--- Episode %d (Initial PCI: %.1f, Age: %.1f) ---", episode, info["pci"], info["road_age"])

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_idx = int(action)
            action_name = RL_ACTION_NAMES.get(action_idx, f"Action {action_idx}")
            actions_taken.append(action_name)

            obs, reward, terminated, truncated, step_info = env.step(action_idx)
            cumulative_reward += reward
            step_count += 1

            logger.info(
                "  Step %2d | Act: %-19s | PCI: %5.1f | Reward: %6.2f | Defects: %s",
                step_count,
                action_name,
                step_info["pci"],
                reward,
                step_info["defects"],
            )

        logger.info(
            "Episode %d Summary -> Cumulative Reward: %.2f | Final PCI: %.1f | Steps: %d",
            episode,
            cumulative_reward,
            obs[0],
            step_count,
        )

        episode_results.append(
            {
                "episode": float(episode),
                "cumulative_reward": float(round(cumulative_reward, 2)),
                "final_pci": float(round(obs[0], 2)),
                "steps": float(step_count),
            }
        )

    logger.info("=" * 65)
    avg_reward = np.mean([res["cumulative_reward"] for res in episode_results])
    logger.info("Evaluation Complete. Average Cumulative Reward: %.2f", avg_reward)
    logger.info("=" * 65)

    return episode_results


def main() -> None:
    """Parses command-line arguments and triggers training and evaluation."""
    parser = argparse.ArgumentParser(description="Train and evaluate DQN Road Maintenance Agent.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=RL_MODEL_PATH,
        help=f"Filepath to save the trained model (default: {RL_MODEL_PATH}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run after training (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the existing model file.",
    )

    args = parser.parse_args()

    if not args.eval_only:
        train_dqn_agent(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            seed=args.seed,
        )

    evaluate_policy(
        model_path=args.save_path,
        num_episodes=args.eval_episodes,
        seed=args.seed + 10,
    )


if __name__ == "__main__":
    main()
