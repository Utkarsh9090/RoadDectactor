"""Pavement Condition Index (PCI) Calculation Engine.

This module converts defect metrics (utility digs, asphalt patches, cracks, potholes)
into standardized civil engineering PCI ratings (0.0 - 100.0), classifies urgency tiers,
and prepares 5-element state space vectors for reinforcement learning inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)
DIG_PENALTY = settings.DIG_PENALTY
PATCH_PENALTY = settings.PATCH_PENALTY
CRACK_PENALTY = settings.CRACK_PENALTY
POTHOLE_PENALTY = settings.POTHOLE_PENALTY
MAX_PCI = settings.MAX_PCI
MIN_PCI = settings.MIN_PCI


# ---------------------------------------------------------------------------
# Data Models (Pydantic)
# ---------------------------------------------------------------------------
class DefectCounts(BaseModel):
    """Container for quantified surface defect occurrences on a road segment."""

    digs: int = Field(ge=0, default=0, description="Count of unsealed utility cuts/excavations")
    patches: int = Field(ge=0, default=0, description="Count of deteriorated asphalt patches")
    cracks: int = Field(ge=0, default=0, description="Count of longitudinal/transverse/alligator cracks")
    potholes: int = Field(ge=0, default=0, description="Count of structural cavities/potholes")


class RoadSegmentState(BaseModel):
    """Comprehensive snapshot of a road segment's physical and analytical condition."""

    segment_id: int = Field(ge=1, description="Sequential 1-indexed identifier of road segment")
    pci: float = Field(ge=0.0, le=100.0, description="Computed Pavement Condition Index (0-100)")
    defects: DefectCounts = Field(description="Aggregated defect metrics for this segment")
    road_age_years: float = Field(ge=0.0, default=0.0, description="Estimated pavement age in years")
    urgency_rating: str = Field(description="Standardized civil engineering intervention tier")


# ---------------------------------------------------------------------------
# Utility & Extraction Helpers
# ---------------------------------------------------------------------------
def extract_defect_counts(defects_input: Union[DefectCounts, Mapping[str, Any]]) -> DefectCounts:
    """Converts a DefectCounts instance or mapping dictionary into a validated DefectCounts object.

    Args:
        defects_input: DefectCounts instance or dict of defect counts.

    Returns:
        DefectCounts: Validated Pydantic model with non-negative integers.
    """
    if isinstance(defects_input, DefectCounts):
        return defects_input

    def _parse(keys: Tuple[str, ...]) -> int:
        for k in keys:
            if k in defects_input:
                try:
                    val = int(defects_input[k])
                    return max(0, val)
                except (ValueError, TypeError):
                    logger.warning("Invalid value for defect key '%s': %s. Defaulting to 0.", k, defects_input[k])
                    return 0
        return 0

    digs = _parse(("digs", "dig", "utility_dig", "utility_digs"))
    patches = _parse(("patches", "patch"))
    cracks = _parse(("cracks", "crack"))
    potholes = _parse(("potholes", "pothole"))

    return DefectCounts(digs=digs, patches=patches, cracks=cracks, potholes=potholes)


def calculate_deduct_value(defects: Union[DefectCounts, Mapping[str, Any]]) -> float:
    """Computes total Deduct Value (DV) according to PRD Section 4 formula:

    DV = (N_digs * 12) + (N_patches * 8) + (N_cracks * 4) + (N_potholes * 15)

    Args:
        defects: DefectCounts instance or defect dictionary.

    Returns:
        float: Total deduct value.
    """
    counts = extract_defect_counts(defects)
    dv = (
        float(counts.digs) * DIG_PENALTY
        + float(counts.patches) * PATCH_PENALTY
        + float(counts.cracks) * CRACK_PENALTY
        + float(counts.potholes) * POTHOLE_PENALTY
    )
    return float(dv)


# ---------------------------------------------------------------------------
# Core PCI Engine Functions
# ---------------------------------------------------------------------------
def calculate_pci(defects: Union[DefectCounts, Mapping[str, Any]]) -> float:
    """Calculates the Pavement Condition Index (PCI) score with strict boundary capping.

    Formula:
        PCI = 100.0 - ((digs * 12) + (patches * 8) + (cracks * 4) + (potholes * 15))
        Hard-capped between 0.0 and 100.0.

    Args:
        defects: DefectCounts instance or mapping of defect counts.

    Returns:
        float: Computed PCI bounded between 0.0 and 100.0, rounded to 2 decimal places.
    """
    try:
        dv = calculate_deduct_value(defects)
        raw_pci = MAX_PCI - dv
        bounded_pci = max(MIN_PCI, min(MAX_PCI, float(raw_pci)))
        return round(bounded_pci, 2)
    except Exception as exc:
        logger.error("PCI calculation failed: %s. Defaulting to 100.0.", exc)
        return float(MAX_PCI)


def classify_urgency(pci: float) -> str:
    """Classifies a PCI score into standardized intervention urgency tiers.

    Tiers:
        - PCI >= 85.0 -> "GOOD"
        - 65.0 <= PCI < 85.0 -> "PREVENTIVE_MONITORING"
        - 40.0 <= PCI < 65.0 -> "MINOR_REPAIR_NEEDED"
        - PCI < 40.0 -> "CRITICAL_RECONSTRUCTION_REQUIRED"

    Args:
        pci: Pavement Condition Index score (0.0 - 100.0).

    Returns:
        str: Standardized urgency classification tier.
    """
    bounded = max(0.0, min(100.0, float(pci)))
    if bounded >= 85.0:
        return "GOOD"
    if bounded >= 65.0:
        return "PREVENTIVE_MONITORING"
    if bounded >= 40.0:
        return "MINOR_REPAIR_NEEDED"
    return "CRITICAL_RECONSTRUCTION_REQUIRED"


def get_pci_condition_category(pci: float) -> str:
    """Backwards-compatible wrapper mapping PCI to legacy condition labels."""
    urgency = classify_urgency(pci)
    mapping = {
        "GOOD": "Good",
        "PREVENTIVE_MONITORING": "Satisfactory",
        "MINOR_REPAIR_NEEDED": "Fair",
        "CRITICAL_RECONSTRUCTION_REQUIRED": "Critical",
    }
    return mapping.get(urgency, "Critical")


def to_rl_state_vector(segment_state: RoadSegmentState) -> np.ndarray:
    """Converts a RoadSegmentState instance into a strict float32 NumPy state vector:

    Format:
        array([pci, digs, patches, cracks, road_age_years], dtype=np.float32)

    Boundary checks ensure values are clipped to the Gymnasium observation space:
        PCI in [0.0, 100.0]
        Digs in [0.0, 50.0]
        Patches in [0.0, 50.0]
        Cracks in [0.0, 50.0]
        Road Age in [0.0, 30.0]

    Args:
        segment_state: RoadSegmentState instance.

    Returns:
        np.ndarray: 1D array of shape (5,) and dtype np.float32.
    """
    clean_pci = float(np.clip(segment_state.pci, 0.0, 100.0))
    clean_digs = float(np.clip(segment_state.defects.digs, 0.0, 50.0))
    clean_patches = float(np.clip(segment_state.defects.patches, 0.0, 50.0))
    clean_cracks = float(np.clip(segment_state.defects.cracks, 0.0, 50.0))
    clean_age = float(np.clip(segment_state.road_age_years, 0.0, 30.0))

    return np.array(
        [clean_pci, clean_digs, clean_patches, clean_cracks, clean_age],
        dtype=np.float32,
    )


def build_rl_state(
    pci: float,
    defects_dict: Union[DefectCounts, Mapping[str, Any]],
    road_age_years: float = 0.0,
) -> np.ndarray:
    """Builds the 5-element RL state vector from raw metrics or mapping.

    Args:
        pci: Current computed Pavement Condition Index.
        defects_dict: Defect metrics as DefectCounts or dictionary.
        road_age_years: Estimated pavement age in years.

    Returns:
        np.ndarray: 1D array of shape (5,) and dtype np.float32.
    """
    counts = extract_defect_counts(defects_dict)
    urgency = classify_urgency(pci)
    state = RoadSegmentState(
        segment_id=1,
        pci=pci,
        defects=counts,
        road_age_years=road_age_years,
        urgency_rating=urgency,
    )
    return to_rl_state_vector(state)


if __name__ == "__main__":
    logger.info("Executing self-test suite for vision/pci_engine.py...")

    # Test 1: Perfect road (0 defects = 100.0 PCI)
    perfect_counts = DefectCounts(digs=0, patches=0, cracks=0, potholes=0)
    p_pci = calculate_pci(perfect_counts)
    assert p_pci == 100.0
    assert classify_urgency(p_pci) == "GOOD"
    logger.info("Test 1 Passed: Perfect Road -> PCI: %.1f, Tier: %s", p_pci, classify_urgency(p_pci))

    # Test 2: Heavily damaged road (caps at 0.0)
    damaged_counts = DefectCounts(digs=10, patches=10, cracks=20, potholes=5)
    d_pci = calculate_pci(damaged_counts)
    assert d_pci == 0.0
    assert classify_urgency(d_pci) == "CRITICAL_RECONSTRUCTION_REQUIRED"
    logger.info("Test 2 Passed: Heavily Damaged -> PCI: %.1f, Tier: %s", d_pci, classify_urgency(d_pci))

    # Test 3: RL State Vector Conversion
    segment = RoadSegmentState(
        segment_id=1,
        pci=p_pci,
        defects=perfect_counts,
        road_age_years=2.5,
        urgency_rating="GOOD",
    )
    vec = to_rl_state_vector(segment)
    assert vec.shape == (5,)
    assert vec.dtype == np.float32
    assert np.allclose(vec, [100.0, 0.0, 0.0, 0.0, 2.5])
    logger.info("Test 3 Passed: RL State Vector -> %s", vec)

    logger.info("All self-tests passed successfully!")
