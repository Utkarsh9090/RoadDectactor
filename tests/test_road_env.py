"""Unit tests for the custom Gymnasium RoadMaintenanceEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from rl_engine.road_env import RoadMaintenanceEnv


def test_environment_spaces() -> None:
    """Validates that observation and action spaces match strict project specs."""
    env = RoadMaintenanceEnv()

    # Observation space: Box(5,)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (5,)
    assert np.allclose(env.observation_space.low, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(env.observation_space.high, [100.0, 50.0, 50.0, 50.0, 30.0])

    # Action space: Discrete(3)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3


def test_reset_default_and_custom() -> None:
    """Tests environment reset behavior under default and custom initial conditions."""
    env = RoadMaintenanceEnv()

    # Default reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert obs[0] == 100.0  # Fresh road has 100 PCI
    assert obs[4] == 0.0    # 0 age

    # Custom reset
    custom_options = {
        "pci": 55.0,
        "digs": 3,
        "patches": 2,
        "cracks": 4,
        "road_age": 7.0,
    }
    obs_custom, info_custom = env.reset(options=custom_options)
    assert np.allclose(obs_custom, [55.0, 3.0, 2.0, 4.0, 7.0])


def test_action_major_resurfacing() -> None:
    """Action 2 (Major Resurfacing) must reset defects to 0 and restore PCI to 100 before degradation."""
    env = RoadMaintenanceEnv()
    env.reset(options={"pci": 20.0, "digs": 5, "patches": 5, "cracks": 10, "road_age": 6.0})

    obs, reward, term, trunc, info = env.step(action=2)

    # Defects reset to 0 (plus minor annual wear)
    assert info["action"] == 2
    assert info["action_cost"] == 60.0
    # PCI was restored to 100, then minus modest annual wear
    assert obs[0] >= 85.0


def test_action_preventive_repair() -> None:
    """Action 1 (Preventive Repair) costs 15 points and provides a +20 PCI boost."""
    env = RoadMaintenanceEnv()
    env.reset(options={"pci": 50.0, "digs": 2, "patches": 2, "cracks": 2, "road_age": 2.0})

    obs, reward, term, trunc, info = env.step(action=1)
    assert info["action"] == 1
    assert info["action_cost"] == 15.0


def test_reward_function_and_penalty() -> None:
    """Reward R = (PCI / 10.0) - C_action - P_failure where P_failure = 50 if PCI < 40."""
    env = RoadMaintenanceEnv()
    # Force low PCI to trigger penalty
    env.reset(options={"pci": 30.0, "digs": 4, "patches": 4, "cracks": 10, "road_age": 8.0})

    obs, reward, term, trunc, info = env.step(action=0)  # Do Nothing (Cost: 0)

    assert info["failure_penalty"] == 50.0
    # Expected reward: (PCI / 10.0) - 0 - 50.0 -> negative value
    expected_reward = (info["pci"] / 10.0) - 0.0 - 50.0
    assert np.isclose(reward, expected_reward, atol=0.01)


def test_episode_truncation_on_max_age() -> None:
    """Environment must truncate when road age reaches 15 years."""
    env = RoadMaintenanceEnv()
    env.reset(options={"pci": 80.0, "digs": 0, "patches": 0, "cracks": 0, "road_age": 14.0})

    obs, reward, term, trunc, info = env.step(action=0)
    assert trunc is True
