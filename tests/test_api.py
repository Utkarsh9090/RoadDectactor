"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from api.main import app
from config import SAMPLE_VIDEOS_DIR

client = TestClient(app)


def test_health_endpoint() -> None:
    """Tests the /health endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "rl_model_available" in data


def test_predict_action_endpoint() -> None:
    """Tests /api/v1/predict-action with sample defect metrics."""
    payload = {
        "defects": {
            "digs": 2,
            "patches": 3,
            "cracks": 4,
            "potholes": 1,
        },
        "road_age_years": 4.0,
    }
    response = client.post("/api/v1/predict-action", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Deduct value: (2*2.0) + (3*1.5) + (4*0.8) + (1*2.4) = 4.0 + 4.5 + 3.2 + 2.4 = 14.1 -> PCI = 85.9
    assert data["pci_score"] == 85.9
    assert data["condition"] == "Good"
    assert data["rl_recommended_action"] in (0, 1, 2)
    assert len(data["state_vector"]) == 5


def test_analyze_video_endpoint() -> None:
    """Tests /api/v1/analyze-video with the generated sample road inspection video."""
    video_path = SAMPLE_VIDEOS_DIR / "sample_road.mp4"
    assert video_path.exists(), f"Sample video not found at {video_path}"

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/v1/analyze-video",
            files={"file": ("sample_road.mp4", f, "video/mp4")},
            params={"road_age_years": 3.0},
        )

    assert response.status_code == 200
    data = response.json()

    # Verify PRD Section 5 schema compliance
    assert data["status"] == "success"
    assert data["filename"] == "sample_road.mp4"
    assert "total_segments_analyzed" in data
    assert "summary" in data
    assert "overall_pci" in data["summary"]
    assert "recommended_action" in data["summary"]
    assert "total_defects" in data["summary"]
    assert isinstance(data["segment_details"], list)
    assert len(data["segment_details"]) > 0

    first_segment = data["segment_details"][0]
    assert "segment_id" in first_segment
    assert "pci_score" in first_segment
    assert "defects" in first_segment
    assert "rl_recommended_action" in first_segment
    assert "action_name" in first_segment


def test_list_videos_endpoint() -> None:
    """Tests the /api/v1/videos catalog endpoint."""
    response = client.get("/api/v1/videos")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    filenames = [v["filename"] for v in data]
    assert "4695859-uhd_3840_2160_30fps.mp4" in filenames
    assert "16433509_2160_3840_60fps.mp4" in filenames
    assert "20258332-hd_1080_1920_30fps.mp4" in filenames


def test_analyze_stored_video_endpoint() -> None:
    """Tests /api/v1/analyze-stored-video with sample_road.mp4."""
    response = client.post(
        "/api/v1/analyze-stored-video",
        params={"video_name": "sample_road.mp4", "road_age_years": 3.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_road.mp4"
    assert "summary" in data
    assert "overall_pci" in data["summary"]
