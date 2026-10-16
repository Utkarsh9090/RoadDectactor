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
from typing import Any, Dict, List, Optional

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
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
        "rl_model_available": rl_model_available,
        "rl_model_path": RL_MODEL_PATH,
    }


@app.post(
    "/api/v1/predict-action",
    response_model=PredictActionResponse,
    tags=["Inference"],
    summary="Predict maintenance action from defect metrics",
)
def predict_action(request: PredictActionRequest) -> PredictActionResponse:
    """Computes PCI and predicts the optimal maintenance intervention for given defect metrics.

    Args:
        request: Defect metrics and road age.

    Returns:
        PredictActionResponse: Computed PCI, state vector, and RL action recommendation.
    """
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
    summary="Analyze road inspection video feed",
)
async def analyze_video(
    file: UploadFile = File(..., description="Target .mp4 or .avi inspection video"),
    road_age_years: float = Query(3.0, ge=0.0, le=30.0, description="Estimated road age in years"),
) -> AnalyzeVideoResponse:
    """Processes uploaded video file and returns PRD Section 5 compliant JSON report.

    Args:
        file: Multipart uploaded video file.
        road_age_years: Estimated pavement age.

    Returns:
        AnalyzeVideoResponse: Standardized inspection report with defect counts, PCI, and RL actions.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video file submitted.")

    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Must be one of {valid_extensions}.",
        )

    # Save uploaded file to temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / file.filename

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detector = get_detector()
        agent = get_rl_agent()

        logger.info("Executing detection on uploaded video: %s", file.filename)
        det_results = detector.process_video(video_path=str(temp_video_path), display=False)

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
            recommended_action = "Major Resurfacing Required"
        elif any(s.rl_recommended_action == 1 for s in segment_details) or overall_pci < 70.0:
            recommended_action = "Preventive Repair Required"
        else:
            recommended_action = "Do Nothing (Good Condition)"

        return AnalyzeVideoResponse(
            status="success",
            filename=file.filename,
            total_segments_analyzed=len(segment_details),
            summary=AnalysisSummary(
                overall_pci=overall_pci,
                recommended_action=recommended_action,
                total_defects=total_defects,
            ),
            segment_details=segment_details,
        )

    except Exception as exc:
        logger.error("Video analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {exc}",
        )
    finally:
        # Clean up temporary upload directory
        shutil.rmtree(temp_dir, ignore_errors=True)
