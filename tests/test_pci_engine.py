"""Unit tests for the Pavement Condition Index (PCI) Engine and State Models."""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError
import pytest
from vision.pci_engine import (
    DefectCounts,
    RoadSegmentState,
    build_rl_state,
    calculate_deduct_value,
    calculate_pci,
    classify_urgency,
    get_pci_condition_category,
    to_rl_state_vector,
)


# ---------------------------------------------------------------------------
# Test Data Models
# ---------------------------------------------------------------------------
def test_defect_counts_defaults_and_validation() -> None:
    """DefectCounts should default to 0 and reject negative integers."""
    counts = DefectCounts()
    assert counts.digs == 0
    assert counts.patches == 0
    assert counts.cracks == 0
    assert counts.potholes == 0

    valid = DefectCounts(digs=2, patches=3, cracks=4, potholes=1)
    assert valid.digs == 2

    # Negative counts must raise ValidationError
    with pytest.raises(ValidationError):
        DefectCounts(digs=-1)
    with pytest.raises(ValidationError):
        DefectCounts(potholes=-5)


def test_road_segment_state_validation() -> None:
    """RoadSegmentState requires valid segment_id, pci in [0, 100], and non-negative age."""
    defects = DefectCounts(digs=1, patches=1, cracks=1, potholes=0)
    state = RoadSegmentState(
        segment_id=1,
        pci=76.0,
        defects=defects,
        road_age_years=3.5,
        urgency_rating="PREVENTIVE_MONITORING",
    )
    assert state.segment_id == 1
    assert state.pci == 76.0
    assert state.road_age_years == 3.5

    # PCI out of range (0-100) must raise ValidationError
    with pytest.raises(ValidationError):
        RoadSegmentState(
            segment_id=1,
            pci=105.0,
            defects=defects,
            road_age_years=1.0,
            urgency_rating="GOOD",
        )


# ---------------------------------------------------------------------------
# Test PCI Calculations
# ---------------------------------------------------------------------------
def test_calculate_pci_perfect_road() -> None:
    """Road with zero defects must yield the maximum PCI score of 100.0."""
    zero_defects = DefectCounts(digs=0, patches=0, cracks=0, potholes=0)
    assert calculate_pci(zero_defects) == 100.0


def test_calculate_pci_heavily_damaged_road() -> None:
    """Road with extensive defects must be hard-capped at 0.0 PCI."""
    heavy_defects = DefectCounts(digs=10, patches=15, cracks=30, potholes=8)
    assert calculate_pci(heavy_defects) == 0.0


def test_calculate_pci_prd_sample_case() -> None:
    """Validates deduct value and PCI against PRD Section 3 & 4 sample:

    Digs: 2 (2*12=24), Patches: 3 (3*8=24), Cracks: 5 (5*4=20), Potholes: 1 (1*15=15)
    Total Deduct Value = 83.0 -> PCI = 100 - 83 = 17.0
    """
    sample_defects = DefectCounts(digs=2, patches=3, cracks=5, potholes=1)
    assert calculate_deduct_value(sample_defects) == 83.0
    assert calculate_pci(sample_defects) == 17.0


def test_calculate_pci_dictionary_input() -> None:
    """PCI engine should accept dictionary input for backward compatibility."""
    dict_input = {"digs": 1, "patches": 1, "cracks": 1, "potholes": 1}
    # 12 + 8 + 4 + 15 = 39 -> PCI = 61.0
    assert calculate_pci(dict_input) == 61.0


# ---------------------------------------------------------------------------
# Test Urgency Classification Tiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pci, expected_urgency",
    [
        (100.0, "GOOD"),
        (85.0, "GOOD"),
        (84.9, "PREVENTIVE_MONITORING"),
        (65.0, "PREVENTIVE_MONITORING"),
        (64.9, "MINOR_REPAIR_NEEDED"),
        (40.0, "MINOR_REPAIR_NEEDED"),
        (39.9, "CRITICAL_RECONSTRUCTION_REQUIRED"),
        (17.0, "CRITICAL_RECONSTRUCTION_REQUIRED"),
        (0.0, "CRITICAL_RECONSTRUCTION_REQUIRED"),
    ],
)
def test_classify_urgency_tiers(pci: float, expected_urgency: str) -> None:
    """Verifies that PCI values map exactly to the specified urgency tiers."""
    assert classify_urgency(pci) == expected_urgency


def test_legacy_get_pci_condition_category() -> None:
    """Verifies backward compatibility mapping to legacy condition categories."""
    assert get_pci_condition_category(90.0) == "Good"
    assert get_pci_condition_category(75.0) == "Satisfactory"
    assert get_pci_condition_category(55.0) == "Fair"
    assert get_pci_condition_category(35.0) == "Critical"


# ---------------------------------------------------------------------------
# Test RL State Vector Conversion
# ---------------------------------------------------------------------------
def test_to_rl_state_vector_shape_and_type() -> None:
    """State vector must be a 1D float32 NumPy array of shape (5,) with order [PCI, Digs, Patches, Cracks, Age]."""
    defects = DefectCounts(digs=2, patches=3, cracks=4, potholes=1)
    state = RoadSegmentState(
        segment_id=1,
        pci=55.5,
        defects=defects,
        road_age_years=4.2,
        urgency_rating="MINOR_REPAIR_NEEDED",
    )
    vec = to_rl_state_vector(state)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (5,)
    assert vec.dtype == np.float32
    assert np.allclose(vec, [55.5, 2.0, 3.0, 4.0, 4.2])


def test_to_rl_state_vector_boundary_clipping() -> None:
    """State vector conversion must clip elements exceeding Gymnasium observation bounds."""
    defects = DefectCounts(digs=80, patches=75, cracks=90, potholes=10)
    state = RoadSegmentState(
        segment_id=1,
        pci=100.0,
        defects=defects,
        road_age_years=45.0,
        urgency_rating="GOOD",
    )
    vec = to_rl_state_vector(state)

    assert vec[0] == 100.0  # PCI max
    assert vec[1] == 50.0   # Digs capped at 50
    assert vec[2] == 50.0   # Patches capped at 50
    assert vec[3] == 50.0   # Cracks capped at 50
    assert vec[4] == 30.0   # Age capped at 30


def test_build_rl_state_convenience_function() -> None:
    """Verifies that build_rl_state correctly wraps metrics into a 5-element state vector."""
    defects = {"digs": 3, "patches": 2, "cracks": 1, "potholes": 0}
    vec = build_rl_state(pci=80.0, defects_dict=defects, road_age_years=2.0)
    assert vec.shape == (5,)
    assert np.allclose(vec, [80.0, 3.0, 2.0, 1.0, 2.0])
