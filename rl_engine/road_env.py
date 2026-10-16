"""Gymnasium Custom Environment for Predictive Road Maintenance Decision Making.

This module formalizes road surface deterioration and maintenance intervention as a
Markov Decision Process (MDP) adhering to Gymnasium specifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure project root is available in sys.path
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
    RL_CRITICAL_PCI_THRESHOLD,
    RL_FAILURE_PENALTY,
    RL_MAX_EPISODE_AGE,
    RL_OBS_HIGH,
    RL_OBS_LOW,
    RL_PREVENTIVE_PCI_BOOST,
    setup_logging,
)

logger = setup_logging(__name__)


class RoadMaintenanceEnv(gym.Env):
    """Custom Gymnasium environment for optimizing predictive road maintenance schedules.

    Observation Space:
        Continuous Box(5) representing [PCI, Digs, Patches, Cracks, Road Age].
        - PCI: [0.0, 100.0]
        - Digs: [0, 50]
        - Patches: [0, 50]
        - Cracks: [0, 50]
        - Road Age: [0, 30] (years)

    Action Space:
        Discrete(3):
        - 0: Do Nothing (Cost: 0, degradation continues based on traffic & age)
        - 1: Preventive Repair (Cost: 15, repairs open digs/patches, restores +20 PCI)
        - 2: Major Resurfacing (Cost: 60, resets defects to 0, restores PCI = 100)

    Reward:
        R = (PCI / 10.0) - C_action - P_failure
        where P_failure = 50.0 if PCI < 40.0 else 0.0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Initializes the road maintenance environment.

        Args:
            render_mode: Optional rendering mode ('human').
        """
        super().__init__()

        self.render_mode: Optional[str] = render_mode

        # Define Observation Space: [PCI, Digs, Patches, Cracks, Road Age]
        self.observation_space: spaces.Box = spaces.Box(
            low=RL_OBS_LOW,
            high=RL_OBS_HIGH,
            dtype=np.float32,
        )

        # Define Action Space: Discrete(3)
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        # Internal state variables
        self.pci: float = 100.0
        self.digs: int = 0
        self.patches: int = 0
        self.cracks: int = 0
        self.road_age: float = 0.0

        logger.debug("RoadMaintenanceEnv initialized with obs_space %s and act_space %s",
                     self.observation_space, self.action_space)

    def _get_obs(self) -> np.ndarray:
        """Constructs and returns the sanitized observation vector.

        Returns:
            np.ndarray: [PCI, Digs, Patches, Cracks, Road Age] in float32.
        """
        obs = np.array(
            [
                np.clip(self.pci, RL_OBS_LOW[0], RL_OBS_HIGH[0]),
                np.clip(self.digs, RL_OBS_LOW[1], RL_OBS_HIGH[1]),
                np.clip(self.patches, RL_OBS_LOW[2], RL_OBS_HIGH[2]),
                np.clip(self.cracks, RL_OBS_LOW[3], RL_OBS_HIGH[3]),
                np.clip(self.road_age, RL_OBS_LOW[4], RL_OBS_HIGH[4]),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the road environment to an initial condition.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dictionary to customize initial state:
                - 'pci': float
                - 'digs': int
                - 'patches': int
                - 'cracks': int
                - 'road_age': float
                - 'randomize': bool (if True, initializes realistic noisy state)

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (initial_observation, info_dictionary)
        """
        super().reset(seed=seed)

        if options and not options.get("randomize", False):
            self.pci = float(np.clip(options.get("pci", 100.0), MIN_PCI, MAX_PCI))
            self.digs = int(np.clip(options.get("digs", 0), 0, 50))
            self.patches = int(np.clip(options.get("patches", 0), 0, 50))
            self.cracks = int(np.clip(options.get("cracks", 0), 0, 50))
            self.road_age = float(np.clip(options.get("road_age", 0.0), 0.0, 30.0))
        elif options and options.get("randomize", False):
            # Realistic random initial state
            self.road_age = float(self.np_random.uniform(0.0, 8.0))
            self.digs = int(self.np_random.integers(0, 4))
            self.patches = int(self.np_random.integers(0, 5))
            self.cracks = int(self.np_random.integers(0, 8))
            # Calculate PCI with some variance
            deduct = (self.digs * 12.0) + (self.patches * 8.0) + (self.cracks * 4.0)
            self.pci = float(np.clip(100.0 - deduct - (self.road_age * 2.0), MIN_PCI, MAX_PCI))
        else:
            # Default brand new road surface
            self.pci = 100.0
            self.digs = 0
            self.patches = 0
            self.cracks = 0
            self.road_age = 0.0

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies a maintenance action, simulates road deterioration, and computes reward.

        Args:
            action: Action integer in {0: Do Nothing, 1: Preventive, 2: Major Resurfacing}.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            logger.warning("Invalid action received: %s. Defaulting to Do Nothing (0).", action)
            action = ACTION_DO_NOTHING

        action_cost = RL_ACTION_COSTS[action]

        # Apply maintenance intervention effects
        if action == ACTION_DO_NOTHING:
            # Natural wear occurs with zero repair cost
            pass
        elif action == ACTION_PREVENTIVE:
            # Preventive maintenance repairs open utility cuts and deteriorated patches
            self.digs = max(0, self.digs - 3)
            self.patches = max(0, self.patches - 2)
            self.cracks = max(0, self.cracks - 1)
            # Restores +20 PCI
            self.pci = min(MAX_PCI, self.pci + RL_PREVENTIVE_PCI_BOOST)
        elif action == ACTION_MAJOR_RESURFACING:
            # Complete road reconstruction resets all defects and restores 100 PCI
            self.digs = 0
            self.patches = 0
            self.cracks = 0
            self.pci = MAX_PCI

        # Simulate annual deterioration dynamics
        self.road_age += 1.0

        # Base wear + defect compounding
        new_cracks = int(self.np_random.integers(0, 3)) if action != ACTION_MAJOR_RESURFACING else 0
        self.cracks = min(50, self.cracks + new_cracks)

        if self.road_age > 3.0 and self.np_random.random() < 0.35:
            self.patches = min(50, self.patches + 1)
        if self.np_random.random() < 0.20:
            self.digs = min(50, self.digs + 1)

        # Deterioration drop in PCI: traffic, age, and existing defect degradation
        defect_decay = (self.digs * 1.5) + (self.patches * 1.0) + (self.cracks * 0.5)
        age_decay = 2.0 + (0.5 * min(10.0, self.road_age))
        total_pci_drop = age_decay + defect_decay

        self.pci = max(MIN_PCI, min(MAX_PCI, self.pci - total_pci_drop))

        # Reward Calculation: R = (PCI / 10.0) - C_action - P_failure
        failure_penalty = RL_FAILURE_PENALTY if self.pci < RL_CRITICAL_PCI_THRESHOLD else 0.0
        reward = float((self.pci / 10.0) - action_cost - failure_penalty)

        # Termination & Truncation criteria
        terminated = bool(self.pci <= MIN_PCI)
        truncated = bool(self.road_age >= RL_MAX_EPISODE_AGE)

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "pci": round(self.pci, 2),
            "action": action,
            "action_name": RL_ACTION_NAMES[action],
            "action_cost": action_cost,
            "failure_penalty": failure_penalty,
            "reward": round(reward, 3),
            "road_age": round(self.road_age, 1),
            "defects": {
                "digs": self.digs,
                "patches": self.patches,
                "cracks": self.cracks,
            },
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Renders the current environment state to the console logger."""
        logger.info(
            "Road Status | Age: %.1f yrs | PCI: %.1f | Digs: %d | Patches: %d | Cracks: %d",
            self.road_age,
            self.pci,
            self.digs,
            self.patches,
            self.cracks,
        )
