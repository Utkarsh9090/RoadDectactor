# AI-Based Smart Road Defect Detection & Predictive Maintenance Platform

An intelligent infrastructure solution that detects road surface defects from video feeds, calculates standardized structural health scores, and uses Reinforcement Learning (RL) to recommend optimal maintenance interventions.

---

## Key Features
* **OpenCV & YOLO Defect Detection:** Real-time processing of dashcam videos to identify utility cuts, patches, cracks, and potholes.
* **Pavement Condition Index (PCI) Engine:** Automated calculation of segment condition scores ($0 - 100$) based on defect severity weights.
* **DQN Predictive Maintenance Policy:** A Deep Q-Network agent trained in a custom Gymnasium environment that determines optimal maintenance actions (*Do Nothing*, *Preventive Repair*, or *Major Resurfacing*).
* **Interactive Web Dashboard:** A Streamlit interface for uploading inspection videos, viewing detection bounding boxes, and analyzing maintenance alerts.
* **FastAPI Microservice Backend:** Programmatic endpoints for integrating analysis outputs with external infrastructure management systems.

---

## Quick Start Guide

### 1. Repository Setup & Virtual Environment

```bash
# Clone repository
git clone https://github.com/your-org/smart-road-detection.git
cd smart-road-detection

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Reinforcement Learning Policy Training

To train the Deep Q-Network (DQN) agent on the custom Gymnasium road environment:

```bash
# Train RL Agent and export model to data/models/dqn_road_policy.zip
python -m rl_engine.train
```

---

### 3. Running the FastAPI Service

Start the backend REST API to process video feeds programmatically:

```bash
# Launch FastAPI server on port 8000
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive API Documentation will be available at: `http://localhost:8000/docs`

---

### 4. Running the Streamlit Web Dashboard

To launch the interactive web application for visual analysis:

```bash
streamlit run dashboard/app.py
```
* Open your browser at `http://localhost:8501`.

---

## Verification & Testing

Execute unit tests to verify PCI calculation engines and Gymnasium environment mechanics:

```bash
# Run test suite
pytest
```

---

## System Architecture Workflow

```text
[ Video Input ] ➔ [ OpenCV Frame Ingestion ] ➔ [ YOLO Defect Detection ]
                                                      │
[ Maintenance Decision ] ◄─ [ DQN RL Agent ] ◄─ [ PCI Calculation Engine ]
```

---

## License & Attribution
Distributed under the MIT License. Built using PyTorch, Ultralytics YOLOv8, Gymnasium, Stable-Baselines3, FastAPI, and Streamlit.