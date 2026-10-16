"""Streamlit Web Dashboard for Smart Road Defect Detection and Predictive Maintenance.

Provides an interactive user interface for uploading dashcam inspection videos,
visualizing real-time defect detections, rendering Plotly PCI gauges and defect charts,
displaying RL policy recommendations, and exporting inspection logs.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional

# Ensure project root is available in sys.path when launched directly via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from stable_baselines3 import DQN
import streamlit as st

from config import (
    PCI_WEIGHTS,
    RL_ACTION_NAMES,
    RL_MODEL_PATH,
    SAMPLE_VIDEOS_DIR,
    setup_logging,
)
from vision.detector import RoadDefectDetector
from vision.pci_engine import build_rl_state, calculate_pci, get_pci_condition_category

logger = setup_logging(__name__)


# Configure Streamlit page
st.set_page_config(
    page_title="AI Smart Road Defect & Maintenance Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphism & Sleek Dark UI)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .alert-green {
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .alert-yellow {
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid #eab308;
        color: #facc15;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .alert-red {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rl_agent() -> Optional[DQN]:
    """Loads and caches the trained Deep Q-Network policy agent."""
    model_path = Path(RL_MODEL_PATH)
    if not model_path.exists():
        logger.warning("RL policy model not found at: %s", model_path)
        return None
    try:
        logger.info("Loading RL agent from %s...", model_path)
        return DQN.load(str(model_path))
    except Exception as exc:
        logger.error("Failed to load RL model: %s", exc)
        return None


@st.cache_resource
def load_detector() -> RoadDefectDetector:
    """Instantiates and caches the computer vision defect detector."""
    return RoadDefectDetector()


def render_pci_gauge(pci_score: float) -> go.Figure:
    """Renders a modern Plotly gauge chart for the Pavement Condition Index.

    Args:
        pci_score: Score between 0.0 and 100.0.

    Returns:
        go.Figure: Interactive Plotly Gauge chart.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pci_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Overall Pavement Condition Index (PCI)", "font": {"size": 20, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#38bdf8", "thickness": 0.28},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.45)"},      # Critical (Red)
                    {"range": [40, 70], "color": "rgba(234, 179, 8, 0.45)"},     # Preventive (Yellow)
                    {"range": [70, 100], "color": "rgba(34, 197, 94, 0.45)"},    # Good (Green)
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.8,
                    "value": pci_score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=280,
        margin={"t": 40, "b": 20, "l": 30, "r": 30},
    )
    return fig


def render_defect_barchart(defects_dict: Dict[str, int]) -> go.Figure:
    """Generates an interactive Plotly horizontal bar chart of defect counts."""
    categories = ["Utility Digs", "Asphalt Patches", "Cracks", "Potholes"]
    counts = [
        defects_dict.get("digs", 0),
        defects_dict.get("patches", 0),
        defects_dict.get("cracks", 0),
        defects_dict.get("potholes", 0),
    ]
    colors = ["#f97316", "#eab308", "#38bdf8", "#ef4444"]

    df = pd.DataFrame({"Defect Type": categories, "Count": counts, "Color": colors})
    fig = px.bar(
        df,
        x="Count",
        y="Defect Type",
        orientation="h",
        text="Count",
        color="Defect Type",
        color_discrete_map={c: col for c, col in zip(categories, colors)},
        title="Defect Occurrence Distribution",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        height=280,
        showlegend=False,
        margin={"t": 40, "b": 20, "l": 20, "r": 20},
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155"},
    )
    fig.update_traces(textposition="outside")
    return fig


def main() -> None:
    """Main Streamlit application entry point."""
    st.markdown('<div class="main-header">🛣️ AI Smart Road Defect Detection & Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Computer Vision (YOLOv8) Pavement Health Scoring & Deep Q-Network Maintenance Scheduling</div>',
        unsafe_allow_html=True,
    )

    rl_model = load_rl_agent()
    detector = load_detector()

    # Sidebar Navigation & Settings
    st.sidebar.title("⚙️ Inspection Settings")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Navigation Mode:",
        ["📹 Video Inspection & Analysis", "🎛️ Interactive What-If Simulator", "ℹ️ System Architecture"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model & Policy Diagnostics")
    st.sidebar.info(
        f"**RL Policy:** {'✅ Loaded (DQN)' if rl_model is not None else '❌ Not Found'}\n\n"
        f"**Vision Engine:** YOLOv8 ({'Road Defect' if detector.has_road_classes else 'Fallback + CV Heuristic'})\n\n"
        f"**Frame Skip:** Every 5th frame\n\n"
        f"**Segment Window:** 100 frames"
    )

    if mode == "📹 Video Inspection & Analysis":
        render_video_inspection_page(detector, rl_model)
    elif mode == "🎛️ Interactive What-If Simulator":
        render_simulator_page(rl_model)
    else:
        render_architecture_page()


def render_video_inspection_page(detector: RoadDefectDetector, rl_model: Optional[DQN]) -> None:
    """Renders the video upload, processing, and visual analysis dashboard."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Source Video Selection")
        source_option = st.radio(
            "Choose Video Source:",
            ["Use Sample Inspection Video (Built-in)", "Upload Custom Video (.mp4 / .avi)"],
        )

        target_video_path: Optional[str] = None
        road_age_val = st.slider("Estimated Road Surface Age (Years):", 0.0, 15.0, 3.5, 0.5)

        if source_option == "Use Sample Inspection Video (Built-in)":
            sample_file = SAMPLE_VIDEOS_DIR / "sample_road.mp4"
            if sample_file.exists():
                target_video_path = str(sample_file)
                st.success(f"Loaded built-in test video: `{sample_file.name}`")
            else:
                st.error("Sample video not found. Please upload a file.")
        else:
            uploaded_file = st.file_uploader("Upload Road Dashcam Video", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                # Save uploaded file to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                target_video_path = tfile.name
                st.success(f"Uploaded: `{uploaded_file.name}`")

        run_btn = st.button("🚀 Run AI Road Inspection", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Video Preview")
        if target_video_path:
            st.video(target_video_path)
        else:
            st.info("Please select or upload a video file to begin analysis.")

    if run_btn and target_video_path:
        with st.spinner("Processing video frames with OpenCV and YOLO..."):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(current: int, total: int) -> None:
                progress_bar.progress(min(1.0, current / max(1, total)))
                status_text.text(f"Processing frame {current} of {total}...")

            try:
                report = detector.process_video(
                    video_path=target_video_path,
                    display=False,
                    progress_callback=progress_cb,
                )
                progress_bar.progress(1.0)
                status_text.text("Inspection complete!")
                st.session_state["inspection_report"] = report
            except Exception as exc:
                st.error(f"Error during video processing: {exc}")
                logger.error("Processing failed: %s", exc)

    # Display results if available in session state
    if "inspection_report" in st.session_state:
        report = st.session_state["inspection_report"]
        st.markdown("---")
        st.subheader("📊 Pavement Inspection Results & Maintenance Directives")

        total_defects = report["summary"]["total_defects"]
        overall_pci = calculate_pci(total_defects)
        condition_str = get_pci_condition_category(overall_pci)

        # RL Inference for each segment
        enriched_segments: List[Dict[str, Any]] = []
        for seg in report["segment_details"]:
            seg_defects = seg["defects"]
            seg_pci = seg["pci_score"]

            if rl_model is not None:
                state_vec = build_rl_state(seg_pci, seg_defects, road_age_val)
                action_idx, _ = rl_model.predict(state_vec, deterministic=True)
                act_int = int(action_idx)
            else:
                act_int = 2 if seg_pci < 40 else (1 if seg_pci < 70 else 0)

            act_name = RL_ACTION_NAMES.get(act_int, "Do Nothing")
            enriched_segments.append({
                "Segment ID": seg["segment_id"],
                "Frame Window": f"{seg['frame_start']} - {seg['frame_end']}",
                "PCI Score": seg_pci,
                "Condition": get_pci_condition_category(seg_pci),
                "Digs": seg_defects["digs"],
                "Patches": seg_defects["patches"],
                "Cracks": seg_defects["cracks"],
                "Potholes": seg_defects["potholes"],
                "RL Action": act_name,
            })

        # Final RL Recommendation
        if any(s["RL Action"] == "Major Resurfacing" for s in enriched_segments) or overall_pci < 40.0:
            rec_action = "Major Resurfacing Required (Immediate Action)"
            alert_class = "alert-red"
            badge = "🔴 CRITICAL"
        elif any(s["RL Action"] == "Preventive Repair" for s in enriched_segments) or overall_pci < 70.0:
            rec_action = "Preventive Repair Required (30-day Window)"
            alert_class = "alert-yellow"
            badge = "🟡 SCHEDULED REPAIR"
        else:
            rec_action = "Do Nothing (Pavement in Optimal Health)"
            alert_class = "alert-green"
            badge = "🟢 OPTIMAL"

        # Alert Banner
        st.markdown(
            f'<div class="{alert_class}">'
            f'<h3>{badge}: {rec_action}</h3>'
            f'<p>Overall Segment PCI Score: <strong>{overall_pci:.1f}/100.0</strong> ({condition_str}) | '
            f'Total Detected Defects: <strong>{sum(total_defects.values())}</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Gauges & Defect charts
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            st.plotly_chart(render_pci_gauge(overall_pci), use_container_width=True)
        with g_col2:
            st.plotly_chart(render_defect_barchart(total_defects), use_container_width=True)

        # Segment Details Table
        st.subheader("📋 Segment-by-Segment Defect & Intervention Breakdown")
        df_segments = pd.DataFrame(enriched_segments)
        st.dataframe(df_segments, use_container_width=True)

        # Export Section
        st.markdown("---")
        st.subheader("📥 Export Inspection Data")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            json_export_data = {
                "status": "success",
                "filename": report.get("filename", "inspection_video.mp4"),
                "total_segments_analyzed": len(enriched_segments),
                "summary": {
                    "overall_pci": overall_pci,
                    "condition": condition_str,
                    "recommended_action": rec_action,
                    "total_defects": total_defects,
                },
                "segment_details": enriched_segments,
            }
            json_str = json.dumps(json_export_data, indent=2)
            st.download_button(
                label="📄 Download Inspection Report (JSON)",
                data=json_str,
                file_name="road_inspection_report.json",
                mime="application/json",
                use_container_width=True,
            )

        with exp_col2:
            csv_buffer = io.StringIO()
            df_segments.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 Download Segment Log (CSV)",
                data=csv_buffer.getvalue(),
                file_name="road_segments_log.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_simulator_page(rl_model: Optional[DQN]) -> None:
    """Interactive What-If Maintenance Intervention Simulator."""
    st.subheader("🎛️ Interactive 'What-If' Defect & Policy Simulator")
    st.markdown("Adjust hypothetical defect numbers and pavement age to simulate PCI scoring and the DQN agent's decision.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Input Defect Parameters")
        digs = st.slider("Unsealed Utility Digs (Weight: -12 PCI)", 0, 10, 2)
        patches = st.slider("Deteriorated Patches (Weight: -8 PCI)", 0, 15, 3)
        cracks = st.slider("Surface Cracks (Weight: -4 PCI)", 0, 25, 4)
        potholes = st.slider("Structural Potholes (Weight: -15 PCI)", 0, 10, 1)
        road_age = st.slider("Road Surface Age (Years)", 0.0, 25.0, 4.0, 0.5)

    defects_dict = {"digs": digs, "patches": patches, "cracks": cracks, "potholes": potholes}
    pci = calculate_pci(defects_dict)
    condition = get_pci_condition_category(pci)
    state_vector = build_rl_state(pci=pci, defects_dict=defects_dict, road_age_years=road_age)

    if rl_model is not None:
        action_idx, _ = rl_model.predict(state_vector, deterministic=True)
        act_int = int(action_idx)
    else:
        act_int = 2 if pci < 40 else (1 if pci < 70 else 0)

    action_name = RL_ACTION_NAMES[act_int]

    with col2:
        st.markdown("#### Real-time AI Policy Output")
        st.plotly_chart(render_pci_gauge(pci), use_container_width=True)

        if act_int == 2:
            st.markdown(
                '<div class="alert-red"><h3>🔴 Action: Major Resurfacing</h3>'
                '<p>Severe pavement degradation detected. Full structural resurfacing recommended.</p></div>',
                unsafe_allow_html=True,
            )
        elif act_int == 1:
            st.markdown(
                '<div class="alert-yellow"><h3>🟡 Action: Preventive Repair</h3>'
                '<p>Moderate wear. Patch sealing and crack filling will restore +20 PCI cost-effectively.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-green"><h3>🟢 Action: Do Nothing</h3>'
                '<p>Pavement condition is optimal. No budget expenditure required at this time.</p></div>',
                unsafe_allow_html=True,
            )

        st.code(
            f"State Vector: {state_vector.tolist()}\n"
            f"Action Code : {act_int} ({action_name})\n"
            f"Condition   : {condition} (PCI {pci}/100)",
            language="python",
        )


def render_architecture_page() -> None:
    """Displays the system design and architecture specifications."""
    st.subheader("ℹ️ System Architecture & Engineering Specifications")
    st.markdown(
        """
        ### Multi-Tier Hybrid AI Architecture

        ```text
        [ Input Video Stream / .mp4 ]
                      │
                      ▼
        ┌───────────────────────────────────────────┐
        │    Module 1: OpenCV & YOLO Processing     │
        │  - Frame Extraction (5-frame skip)        │
        │  - Bounding Box & Class Detection        │
        │  - Tracking & Defect Counting Engine     │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │     Module 2: Road State Aggregator       │
        │  - PCI Score Calculation Engine           │
        │  - State Vector Formatting [5-dim]        │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │   Module 3: RL Maintenance Decision Engine│
        │  - Gymnasium Environment Sync             │
        │  - DQN Inference Agent (Discrete(3))      │
        │  - Action & Risk Projection Output        │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │     Module 4: API & Visual Dashboard      │
        │  - FastAPI Endpoints (/api/v1/...)        │
        │  - Streamlit Control Center UI            │
        └───────────────────────────────────────────┘
        ```

        ### Mathematical Formulations

        1. **Pavement Condition Index (PCI) Formula:**
           $$\\text{DV} = (N_{\\text{digs}} \\times 12) + (N_{\\text{patches}} \\times 8) + (N_{\\text{cracks}} \\times 4) + (N_{\\text{potholes}} \\times 15)$$
           $$\\text{PCI} = \\max(0.0, \\min(100.0, 100.0 - \\text{DV}))$$

        2. **Gymnasium State Vector Space:**
           $$S = [\\text{PCI}, N_{\\text{digs}}, N_{\\text{patches}}, N_{\\text{cracks}}, \\text{Road Age}]$$

        3. **Reinforcement Learning Reward Function:**
           $$R = \\left(\\frac{\\text{PCI}}{10.0}\\right) - C_{\\text{action}} - P_{\\text{failure}}$$
           Where $C_{\\text{action}} \\in \\{0, 15, 60\\}$ and $P_{\\text{failure}} = 50$ if $\\text{PCI} < 40$.
        """
    )


if __name__ == "__main__":
    main()
