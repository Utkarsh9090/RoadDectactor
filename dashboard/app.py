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
    .alert-orange {
        background-color: rgba(249, 115, 22, 0.15);
        border: 1px solid #f97316;
        color: #fb923c;
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
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.45)"},      # Repair Immediate (Red)
                    {"range": [40, 60], "color": "rgba(249, 115, 22, 0.45)"},    # Repair It (Orange)
                    {"range": [60, 80], "color": "rgba(234, 179, 8, 0.45)"},     # Maintain Properly (Yellow)
                    {"range": [80, 100], "color": "rgba(34, 197, 94, 0.45)"},    # Good Condition (Green)
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


def get_available_stored_videos() -> Dict[str, Dict[str, Any]]:
    """Discovers stored real project videos and benchmark samples."""
    catalog: Dict[str, Dict[str, Any]] = {}
    real_vids = {
        "4695859-uhd_3840_2160_30fps.mp4": {
            "title": "🌲 Scenic Paved Forest Highway (4K UHD)",
            "desc": "3840x2160, 30fps, 41.1s (1231 frames, 121 MB) • Smooth asphalt corridor",
            "path": PROJECT_ROOT / "videos" / "4695859-uhd_3840_2160_30fps.mp4",
        },
        "16433509_2160_3840_60fps.mp4": {
            "title": "⛰️ Mountain Pass Trail & Ridge (4K 60fps)",
            "desc": "2160x3840, 60fps, 16.6s (996 frames, 96 MB) • High elevation dynamic",
            "path": PROJECT_ROOT / "videos" / "16433509_2160_3840_60fps.mp4",
        },
        "20258332-hd_1080_1920_30fps.mp4": {
            "title": "🏙️ Urban Arterial Dashcam (1080p Vertical)",
            "desc": "1080x1920, 30fps, 43.6s (1308 frames, 28 MB) • Multi-lane urban pavement with joints",
            "path": PROJECT_ROOT / "videos" / "20258332-hd_1080_1920_30fps.mp4",
        },
    }
    for fname, info in real_vids.items():
        if info["path"].exists():
            catalog[info["title"]] = info

    if SAMPLE_VIDEOS_DIR.exists():
        for p in SAMPLE_VIDEOS_DIR.glob("*.mp4"):
            key = f"🎞️ Sample: {p.name}"
            if key not in catalog:
                catalog[key] = {
                    "title": key,
                    "desc": f"Built-in sample ({round(p.stat().st_size / (1024*1024), 2)} MB)",
                    "path": p,
                }
    return catalog


def render_video_inspection_page(detector: RoadDefectDetector, rl_model: Optional[DQN]) -> None:
    """Renders the video upload, processing, and visual analysis dashboard."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Source Video Selection")
        source_option = st.radio(
            "Choose Video Source:",
            [
                "🎬 Real Highway & Urban Videos (Project Dataset)",
                "📁 Upload Video from System (.mp4 / .mov / .avi)",
                "📹 Record Video Live via Camera (Real-Time)",
                "📸 Snap Instant Photo via Camera",
            ],
        )

        target_video_path: Optional[str] = None
        target_video_filename: Optional[str] = None
        use_cached: bool = False
        road_age_val = st.slider("Estimated Road Surface Age (Years):", 0.0, 15.0, 3.5, 0.5)

        if source_option == "🎬 Real Highway & Urban Videos (Project Dataset)":
            catalog = get_available_stored_videos()
            if catalog:
                selected_title = st.selectbox(
                    "Select Real Road Corridor Video:",
                    options=list(catalog.keys()),
                )
                chosen = catalog[selected_title]
                target_video_path = str(chosen["path"])
                target_video_filename = chosen["path"].name
                st.caption(f"ℹ️ *{chosen['desc']}*")
                use_cached = st.checkbox(
                    "⚡ Use pre-computed AI inspection cache (Instant Result)",
                    value=True,
                    help="Instantly loads high-precision frame detections and evaluates with the trained DQN policy.",
                )
            else:
                st.warning("No stored videos found in videos/ directory.")

        elif source_option == "📁 Upload Video from System (.mp4 / .mov / .avi)":
            st.caption("💻 *Upload inspection video from your local computer or phone storage.*")
            uploaded_file = st.file_uploader("Select Road Inspection Video", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                target_video_path = tfile.name
                st.success(f"Loaded: `{uploaded_file.name}`")

        elif source_option == "📹 Record Video Live via Camera (Real-Time)":
            st.caption("📹 *Record live inspection video with real-time viewfinder display.*")
            cam_index = st.number_input("Camera Index (0 for built-in webcam, 1 for external USB):", min_value=0, max_value=5, value=0)
            rec_seconds = st.slider("Recording Duration (seconds):", 2, 10, 4)
            if st.button("🔴 Start Live Camera Recording", type="primary"):
                cap = cv2.VideoCapture(int(cam_index))
                if not cap.isOpened():
                    st.error(f"Could not open camera device {cam_index}. Check device connection and permissions.")
                else:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    fps = 30
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(tfile.name, fourcc, fps, (width, height))
                    total_frames = rec_seconds * fps

                    preview_box = st.empty()
                    progress_rec = st.progress(0.0)

                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        writer.write(frame)
                        if i % 2 == 0:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            preview_box.image(rgb_frame, caption=f"🔴 Recording in Real-Time... Frame {i+1}/{total_frames}", use_container_width=True)
                            progress_rec.progress((i + 1) / total_frames)

                    cap.release()
                    writer.release()
                    st.session_state["recorded_clip_path"] = tfile.name
                    st.success(f"✅ Successfully recorded {rec_seconds}s clip ({total_frames} frames)!")

            if "recorded_clip_path" in st.session_state:
                target_video_path = st.session_state["recorded_clip_path"]

        elif source_option == "Snap Instant Photo via Camera":
            st.caption("📸 Take an instant photo using your phone or laptop camera for single-frame defect analysis.")
            cam_photo = st.camera_input("Capture road surface photo")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                with st.spinner("Analyzing photo with OpenCV and YOLO..."):
                    annotated_img, counts, _ = detector.detect_frame(img, frame_idx=1)
                    pci = calculate_pci(counts)
                    cat = get_pci_condition_category(pci)
                    state_v = build_rl_state(pci, counts, road_age_val)
                    if rl_model is not None:
                        act_idx, _ = rl_model.predict(state_v, deterministic=True)
                        act_int = int(act_idx)
                    else:
                        act_int = 2 if pci < 40 else (1 if pci < 80 else 0)
                    act_name = RL_ACTION_NAMES.get(act_int, "Do Nothing")

                    st.session_state["photo_results"] = {
                        "annotated_rgb": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        "counts": counts,
                        "pci": pci,
                        "category": cat,
                        "action_name": act_name,
                        "action_int": act_int,
                    }

        if source_option != "Snap Instant Photo via Camera":
            run_btn = st.button("🚀 Run AI Road Inspection", type="primary", use_container_width=True)
        else:
            run_btn = False

    with col2:
        st.subheader("2. Video / Camera Preview")
        if source_option == "Snap Instant Photo via Camera" and "photo_results" in st.session_state:
            res = st.session_state["photo_results"]
            st.image(res["annotated_rgb"], caption=f"Analyzed Road Photo (PCI: {res['pci']:.1f}/100)", use_container_width=True)
            if res["pci"] < 40.0:
                st.markdown(f'<div class="alert-red"><h3>🔴 Action: Repair Immediate</h3><p>Critical structural damage detected (PCI: {res["pci"]:.1f}/100)</p></div>', unsafe_allow_html=True)
            elif res["pci"] < 60.0:
                st.markdown(f'<div class="alert-orange"><h3>🟠 Action: Repair It</h3><p>Significant distress detected (PCI: {res["pci"]:.1f}/100)</p></div>', unsafe_allow_html=True)
            elif res["pci"] < 80.0:
                st.markdown(f'<div class="alert-yellow"><h3>🟡 Action: Maintain Properly</h3><p>Preventive maintenance recommended (PCI: {res["pci"]:.1f}/100)</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-green"><h3>🟢 Action: Good Condition</h3><p>Surface in good condition (PCI: {res["pci"]:.1f}/100)</p></div>', unsafe_allow_html=True)
            st.metric("Detected Defect Occurrences", f"Digs: {res['counts'].get('digs',0)} | Patches: {res['counts'].get('patches',0)} | Cracks: {res['counts'].get('cracks',0)} | Potholes: {res['counts'].get('potholes',0)}")
        elif target_video_path:
            st.video(target_video_path)
        else:
            st.info("Select a video, record a clip, or snap a camera photo to begin analysis.")

    if run_btn and target_video_path:
        loaded_from_cache = False
        if source_option == "🎬 Real Highway & Urban Videos (Project Dataset)" and use_cached and target_video_filename:
            dataset_file = PROJECT_ROOT / "data" / "real_video_dataset.json"
            if dataset_file.exists():
                try:
                    with open(dataset_file, "r", encoding="utf-8") as f:
                        ds = json.load(f)
                    for v in ds.get("videos", []):
                        if v["filename"] == target_video_filename:
                            st.session_state["inspection_report"] = {
                                "summary": v.get("summary", {}),
                                "segment_details": v.get("segment_details", []),
                            }
                            st.success(f"⚡ Loaded pre-computed AI inspection for `{target_video_filename}` ({len(v.get('segment_details', []))} segments)")
                            loaded_from_cache = True
                            break
                except Exception as e:
                    logger.debug("Cache read error: %s", e)

        if not loaded_from_cache:
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
                act_int = 2 if seg_pci < 40 else (1 if seg_pci < 80 else 0)

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

        # Final Recommendation
        if overall_pci < 40.0 or any(s["RL Action"] == "Major Resurfacing" for s in enriched_segments):
            rec_action = "Repair Immediate (Emergency Resurfacing Required)"
            alert_class = "alert-red"
            badge = "🔴 REPAIR IMMEDIATE"
        elif overall_pci < 60.0:
            rec_action = "Repair It (Corrective Surface Repair Required)"
            alert_class = "alert-orange"
            badge = "🟠 REPAIR IT"
        elif overall_pci < 80.0 or any(s["RL Action"] == "Preventive Repair" for s in enriched_segments):
            rec_action = "Maintain Properly (Preventive Maintenance Required)"
            alert_class = "alert-yellow"
            badge = "🟡 MAINTAIN PROPERLY"
        else:
            rec_action = "Good Condition (Routine Monitoring Only)"
            alert_class = "alert-green"
            badge = "🟢 GOOD CONDITION"

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
        digs = st.slider("Unsealed Utility Digs (Weight: -2.0 PCI)", 0, 10, 1)
        patches = st.slider("Deteriorated Patches (Weight: -1.5 PCI)", 0, 15, 1)
        cracks = st.slider("Surface Cracks (Weight: -0.8 PCI)", 0, 25, 5)
        potholes = st.slider("Structural Potholes (Weight: -2.4 PCI)", 0, 10, 1)
        road_age = st.slider("Road Surface Age (Years)", 0.0, 25.0, 4.0, 0.5)

    defects_dict = {"digs": digs, "patches": patches, "cracks": cracks, "potholes": potholes}
    pci = calculate_pci(defects_dict)
    condition = get_pci_condition_category(pci)
    state_vector = build_rl_state(pci=pci, defects_dict=defects_dict, road_age_years=road_age)

    if rl_model is not None:
        action_idx, _ = rl_model.predict(state_vector, deterministic=True)
        act_int = int(action_idx)
    else:
        act_int = 2 if pci < 40 else (1 if pci < 80 else 0)

    action_name = RL_ACTION_NAMES.get(act_int, "Do Nothing")

    with col2:
        st.markdown("#### Real-time AI Policy Output")
        st.plotly_chart(render_pci_gauge(pci), use_container_width=True)

        if pci < 40.0:
            st.markdown(
                '<div class="alert-red"><h3>🔴 Action: Repair Immediate</h3>'
                '<p>Severe pavement degradation detected (&lt; 40 PCI). Full structural resurfacing and immediate repair required.</p></div>',
                unsafe_allow_html=True,
            )
        elif pci < 60.0:
            st.markdown(
                '<div class="alert-orange"><h3>🟠 Action: Repair It</h3>'
                '<p>Substantial distress detected (40–60 PCI). Timely corrective surface repair and patching needed.</p></div>',
                unsafe_allow_html=True,
            )
        elif pci < 80.0:
            st.markdown(
                '<div class="alert-yellow"><h3>🟡 Action: Maintain Properly</h3>'
                '<p>Moderate wear detected (60–80 PCI). Scheduled crack sealing and preventive maintenance to maintain properly.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-green"><h3>🟢 Action: Good Condition</h3>'
                '<p>Pavement condition is optimal (80–100 PCI). Road is in good health, routine monitoring only.</p></div>',
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
