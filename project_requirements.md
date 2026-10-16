# Technical Requirements & Dependencies

## 1. System Requirements
* **Operating System:** Ubuntu 22.04 LTS, macOS (Apple Silicon Supported), or Windows 11 with WSL2.
* **Python Runtime:** Python 3.10.x or Python 3.11.x (Python 3.12+ not recommended due to PyTorch wheel compatibility).
* **GPU Hardware (Optional):** NVIDIA GPU with CUDA 11.8+ capability for hardware acceleration.
* **RAM:** Minimum 8 GB (16 GB recommended for video frame buffer processing).

---

## 2. Python Package Dependencies (`requirements.txt`)

```text
# Computer Vision & Image Processing
opencv-python-headless>=4.8.0.76
ultralytics>=8.1.0
torch>=2.1.0
torchvision>=0.16.0
pillow>=10.0.0

# Reinforcement Learning & Math
gymnasium>=0.29.0
stable-baselines3>=2.2.1
numpy>=1.24.0
pandas>=2.0.0

# Backend API & Server
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Web Dashboard & Visualization
streamlit>=1.30.0
plotly>=5.18.0
matplotlib>=3.8.0

# Utilities
pyyaml>=6.0.1
requests>=2.31.0
pytest>=8.0.0
```

---

## 3. Directory Structure Layout

```text
smart_road_detection/
├── config.py                  # Global configurations and path constants
├── requirements.txt           # Dependency management
├── README.md                  # Project overview and setup commands
├── PRD.md                     # Product Requirement Document
├── REQUIREMENTS.md            # System requirements and specifications
├── data/
│   ├── models/                # Saved YOLO & RL (.zip) models
│   └── sample_videos/         # Test input video files
├── vision/
│   ├── __init__.py
│   ├── detector.py            # OpenCV + YOLO detection logic
│   └── pci_engine.py          # PCI calculation engine
├── rl_engine/
│   ├── __init__.py
│   ├── road_env.py            # Gymnasium custom environment
│   └── train.py               # Stable-baselines3 training script
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI server implementation
└── dashboard/
    └── app.py                 # Streamlit web UI application
```

---

## 4. Dataset Specifications for YOLO Model Training
To train or fine-tune the YOLO object detection model for this project, the dataset must follow the standard YOLO PyTorch annotation format (`.txt` files per image).

* **Class Labels:**
  * `0`: `dig` (unsealed utility cuts)
  * `1`: `patch` (road surface patches)
  * `2`: `crack` (surface cracking)
  * `3`: `pothole` (structural cavity)
* **Dataset Config File (`road_data.yaml`):**
```yaml
path: ../data/dataset
train: images/train
val: images/val

names:
  0: dig
  1: patch
  2: crack
  3: pothole
```

---

## 5. Environment Variables (`.env`)

```ini
# Path Configurations
YOLO_MODEL_PATH=data/models/yolov8_road.pt
RL_MODEL_PATH=data/models/dqn_road_policy.zip
OUTPUT_DIR=data/outputs

# Performance Tuning
FRAME_SKIP_RATE=5
CONFIDENCE_THRESHOLD=0.40
IOU_THRESHOLD=0.45

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
```