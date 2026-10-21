"""FastAPI Microservice Backend for Smart Road Defect Detection and Maintenance Decisions.

Exposes RESTful endpoints for video analysis, PCI score calculation,
and Reinforcement Learning predictive maintenance action inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
from pydantic import BaseModel, Field
from stable_baselines3 import DQN

from config import (
    API_HOST,
    API_PORT,
    RL_ACTION_NAMES,
    RL_MODEL_PATH,
    setup_logging,
)
from vision.detector import RoadDefectDetector
from vision.pci_engine import build_rl_state, calculate_pci, get_pci_condition_category

logger = setup_logging(__name__)


# FastAPI Application Instance
app = FastAPI(
    title="AI Smart Road Defect Detection & Predictive Maintenance API",
    description="Automated pavement inspection, PCI computation, and RL maintenance scheduling.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard and external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
detector_instance: Optional[RoadDefectDetector] = None
rl_agent_instance: Optional[DQN] = None


def get_detector() -> RoadDefectDetector:
    """Lazy loader for RoadDefectDetector."""
    global detector_instance
    if detector_instance is None:
        logger.info("Initializing RoadDefectDetector service...")
        detector_instance = RoadDefectDetector()
    return detector_instance


def get_rl_agent() -> DQN:
    """Lazy loader for trained DQN agent."""
    global rl_agent_instance
    if rl_agent_instance is None:
        model_path = Path(RL_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Trained RL model not found at {model_path}. Run 'python -m rl_engine.train' first."
            )
        logger.info("Loading DQN policy model from: %s", model_path)
        rl_agent_instance = DQN.load(str(model_path))
    return rl_agent_instance


# ---------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------
class DefectCountsModel(BaseModel):
    digs: int = Field(ge=0, default=0, description="Count of unsealed utility cuts")
    patches: int = Field(ge=0, default=0, description="Count of deteriorated asphalt patches")
    cracks: int = Field(ge=0, default=0, description="Count of longitudinal/alligator cracks")
    potholes: int = Field(ge=0, default=0, description="Count of structural road cavities")


class PredictActionRequest(BaseModel):
    defects: DefectCountsModel = Field(description="Defect counts for the road segment")
    road_age_years: float = Field(ge=0.0, le=30.0, default=3.0, description="Road age in years")


class PredictActionResponse(BaseModel):
    status: str
    pci_score: float
    condition: str
    state_vector: List[float]
    rl_recommended_action: int
    action_name: str
    guidance: str


class SegmentDetail(BaseModel):
    segment_id: int
    pci_score: float
    defects: Dict[str, int]
    rl_recommended_action: int
    action_name: str


class AnalysisSummary(BaseModel):
    overall_pci: float
    recommended_action: str
    total_defects: Dict[str, int]


class AnalyzeVideoResponse(BaseModel):
    status: str
    filename: str
    total_segments_analyzed: int
    summary: AnalysisSummary
    segment_details: List[SegmentDetail]


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint providing runtime and model status."""
    rl_model_available = Path(RL_MODEL_PATH).exists()
    return {
        "status": "healthy",
        "service": "Smart Road Defect Detection & Maintenance Platform",
        "vision_detector_loaded": True,
        "rl_policy_loaded": rl_model_available,
        "rl_model_available": rl_model_available,
        "rl_model_path": RL_MODEL_PATH,
    }


def find_stored_video(filename: str) -> Optional[Path]:
    """Searches for a video file across project video repositories."""
    search_dirs = [
        PROJECT_ROOT / "videos",
        PROJECT_ROOT / "data" / "videos",
        PROJECT_ROOT / "data" / "sample_videos",
        PROJECT_ROOT / "data" / "test_clean",
        PROJECT_ROOT / "data" / "test_damaged",
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


VIDEO_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "4695859-uhd_3840_2160_30fps.mp4": {
        "title": "Scenic Forest Highway (4K UHD)",
        "description": "High-resolution paved asphalt highway through mountain forest with dappled canopy lighting and smooth road geometry.",
        "category": "Paved Highway (Rural)",
        "tags": ["4K UHD", "30 FPS", "Landscape 16:9", "Forest Highway", "Dappled Lighting"],
    },
    "16433509_2160_3840_60fps.mp4": {
        "title": "Mountain Rocky Trail (4K 60fps)",
        "description": "High-altitude mountain unpaved trail with rough rocks, loose gravel, and extreme natural terrain.",
        "category": "Unpaved / Extreme Terrain",
        "tags": ["4K UHD", "60 FPS", "Portrait 9:16", "Off-Road", "Rocky Mountain"],
    },
    "20258332-hd_1080_1920_30fps.mp4": {
        "title": "Urban Arterial Dashcam (1080p)",
        "description": "Real vehicle dashboard camera feed on a multi-lane urban road with road signs, streetlights, and lane dividers.",
        "category": "Urban Highway (Dashcam)",
        "tags": ["1080p HD", "30 FPS", "Portrait 9:16", "Urban Dashcam", "Lane Dividers"],
    },
}


@app.get("/api/v1/videos", tags=["Inspection"], summary="List available server inspection videos")
def list_available_videos() -> List[Dict[str, Any]]:
    """Returns catalog of stored inspection videos available on the server."""
    catalog: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    search_dirs = [
        PROJECT_ROOT / "videos",
        PROJECT_ROOT / "data" / "videos",
        PROJECT_ROOT / "data" / "sample_videos",
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            if p.name in seen:
                continue
            seen.add(p.name)

            desc_info = VIDEO_DESCRIPTIONS.get(
                p.name,
                {
                    "title": p.stem.replace("_", " ").title(),
                    "description": f"Stored inspection video file ({p.name})",
                    "category": "Inspection Feed",
                    "tags": ["Inspection Video"],
                },
            )

            # Extract basic video metrics via OpenCV
            cap = cv2.VideoCapture(str(p))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            duration = round(total_frames / max(1.0, fps), 1)
            size_mb = round(p.stat().st_size / (1024 * 1024), 2)

            catalog.append({
                "id": p.name,
                "filename": p.name,
                "title": desc_info["title"],
                "description": desc_info["description"],
                "category": desc_info["category"],
                "tags": desc_info.get("tags", []),
                "resolution": f"{w}x{h}",
                "fps": round(fps, 1),
                "duration_seconds": duration,
                "total_frames": total_frames,
                "size_mb": size_mb,
                "stream_url": f"/api/v1/videos/{p.name}/stream",
            })

    return catalog


@app.get("/api/v1/videos/{filename}/stream", tags=["Inspection"], summary="Stream stored video file")
def stream_video(filename: str):
    """Streams a stored video file for browser video playback."""
    video_path = find_stored_video(filename)
    if not video_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video '{filename}' not found.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename,
    )


def _execute_video_analysis(video_path: Path, filename: str, road_age_years: float) -> AnalyzeVideoResponse:
    """Core executor for defect extraction, PCI scoring, and RL action inference."""
    detector = get_detector()
    agent = get_rl_agent()

    logger.info("Executing detection on video: %s (%s)", filename, video_path)
    det_results = detector.process_video(video_path=str(video_path), display=False)

    segment_details: List[SegmentDetail] = []
    for seg in det_results["segment_details"]:
        seg_defects = seg["defects"]
        seg_pci = seg["pci_score"]

        state_vec = build_rl_state(
            pci=seg_pci,
            defects_dict=seg_defects,
            road_age_years=road_age_years,
        )
        action_idx, _ = agent.predict(state_vec, deterministic=True)
        act_int = int(action_idx)
        act_name = RL_ACTION_NAMES.get(act_int, "Do Nothing")

        segment_details.append(
            SegmentDetail(
                segment_id=seg["segment_id"],
                pci_score=seg_pci,
                defects=seg_defects,
                rl_recommended_action=act_int,
                action_name=act_name,
            )
        )

    total_defects = det_results["summary"]["total_defects"]
    overall_pci = calculate_pci(total_defects)

    if any(s.rl_recommended_action == 2 for s in segment_details) or overall_pci < 40.0:
        recommended_action = "Repair Immediate (Major Resurfacing Required)"
    elif overall_pci < 60.0:
        recommended_action = "Repair It (Corrective Repair Required)"
    elif overall_pci < 80.0:
        recommended_action = "Maintain Properly (Preventive Maintenance)"
    else:
        recommended_action = "Good Condition (Optimal Pavement)"

    return AnalyzeVideoResponse(
        status="success",
        filename=filename,
        total_segments_analyzed=len(segment_details),
        summary=AnalysisSummary(
            overall_pci=overall_pci,
            recommended_action=recommended_action,
            total_defects=total_defects,
        ),
        segment_details=segment_details,
    )


@app.post(
    "/api/v1/analyze-stored-video",
    response_model=AnalyzeVideoResponse,
    tags=["Inspection"],
    summary="Analyze a stored video file on the server without uploading",
)
def analyze_stored_video(
    video_name: str = Query(..., description="Filename of stored video in videos/ or data/videos/"),
    road_age_years: float = Query(3.0, ge=0.0, le=30.0, description="Estimated pavement age in years"),
) -> AnalyzeVideoResponse:
    """Analyzes a stored video on the server directly from local filesystem."""
    video_path = find_stored_video(video_name)
    if not video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stored video '{video_name}' not found on server.",
        )
    try:
        return _execute_video_analysis(video_path, video_name, road_age_years)
    except Exception as exc:
        logger.error("Stored video analysis failed for %s: %s", video_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        )


@app.post(
    "/api/v1/predict-action",
    response_model=PredictActionResponse,
    tags=["Inference"],
    summary="Predict maintenance action from defect metrics",
)
def predict_action(request: PredictActionRequest) -> PredictActionResponse:
    """Computes PCI and predicts the optimal maintenance intervention for given defect metrics."""
    try:
        agent = get_rl_agent()
        defects_dict = request.defects.model_dump()

        pci = calculate_pci(defects_dict)
        condition = get_pci_condition_category(pci)
        state_vec = build_rl_state(pci=pci, defects_dict=defects_dict, road_age_years=request.road_age_years)

        action_idx, _ = agent.predict(state_vec, deterministic=True)
        action_int = int(action_idx)
        action_name = RL_ACTION_NAMES.get(action_int, "Do Nothing")

        if action_int == 0:
            guidance = "Optimal Condition - No Immediate Action Required"
        elif action_int == 1:
            guidance = "Preventive Repair Required (30-day maintenance window)"
        else:
            guidance = "Major Resurfacing Required (Immediate reconstruction priority)"

        return PredictActionResponse(
            status="success",
            pci_score=pci,
            condition=condition,
            state_vector=state_vec.tolist(),
            rl_recommended_action=action_int,
            action_name=action_name,
            guidance=guidance,
        )
    except Exception as exc:
        logger.error("Prediction endpoint failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {exc}",
        )


@app.post(
    "/api/v1/analyze-video",
    response_model=AnalyzeVideoResponse,
    tags=["Inspection"],
    summary="Analyze road inspection video feed via upload",
)
async def analyze_video(
    file: UploadFile = File(..., description="Target .mp4 or .avi inspection video"),
    road_age_years: float = Query(3.0, ge=0.0, le=30.0, description="Estimated road age in years"),
) -> AnalyzeVideoResponse:
    """Processes uploaded video file and returns standardized inspection report."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video file submitted.")

    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Must be one of {valid_extensions}.",
        )

    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / file.filename

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return _execute_video_analysis(temp_video_path, file.filename, road_age_years)

    except Exception as exc:
        logger.error("Uploaded video analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
