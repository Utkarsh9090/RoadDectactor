# Product Requirement Document (PRD)

## 1. Document Overview
* **Project Name:** AI-Based Smart Road Defect Detection & Predictive Maintenance Platform
* **Version:** 1.0.0
* **Status:** Draft / Technical Specification
* **Target Users:** Municipal Authorities, Highway Maintenance Departments, Infrastructure Civil Engineers

---

## 2. Executive Summary
The Smart Road Defect Detection & Predictive Maintenance Platform automates road surface analysis and maintenance scheduling. Using dashcam or video inputs, the system uses OpenCV and YOLO object detection models to identify defects—specifically unsealed utility digs, asphalt patches, alligator cracking, and potholes. 

Unlike standard detection applications, this platform integrates a custom **Pavement Condition Index (PCI) Engine** with a **Reinforcement Learning (RL) Policy Agent** (trained using Deep Q-Networks). The RL agent balances physical deterioration against budget constraints to output optimal maintenance interventions (Do Nothing, Preventive Repair, or Major Resurfacing).

---

## 3. High-Level System Architecture

```
                                [ Input Video Stream / .mp4 ]
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │    Module 1: OpenCV & YOLO Processing     │
                        │  - Frame Extraction (N-frame skip)        │
                        │  - Bounding Box & Class Detection        │
                        │  - Tracking & Defect Counting Engine     │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │     Module 2: Road State Aggregator       │
                        │  - PCI Score Calculation Engine           │
                        │  - State Vector Formatting                │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │   Module 3: RL Maintenance Decision Engine│
                        │  - Gymnasium Environment Sync             │
                        │  - DQN / PPO Inference Agent              │
                        │  - Action & Risk Projection Output        │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │     Module 4: API & Visual Dashboard      │
                        │  - FastAPI Endpoints                      │
                        │  - Interactive Streamlit Dashboard       │
                        └───────────────────────────────────────────┘
```

---

## 4. Technical Module Specifications

### Module 1: OpenCV & Computer Vision Pipeline (`/vision`)
* **Frameworks:** OpenCV (`cv2`), Ultralytics YOLOv8 / YOLOv11.
* **Target Classes:**
  * `0`: Utility Dig (Unsealed/Excavated cut)
  * `1`: Patch (Deteriorated/Uneven patch)
  * `2`: Crack (Longitudinal, Transverse, or Alligator)
  * `3`: Pothole (Open structural cavity)
* **Processing Rules:**
  * Process every 5th frame (`FRAME_SKIP = 5`) to optimize real-time inference latency.
  * Implement non-maximum suppression (NMS) with an IoU threshold of 0.45 to prevent duplicate counts across adjacent frames.
  * Aggregate raw defect counts over 100-frame processing windows (defined as 1 Road Segment).

### Module 2: Pavement Condition Index (PCI) Engine (`/vision/pci_engine.py`)
* **Logic:** Computes a standardized structural health score ($0.0$ to $100.0$) per segment.
* **Formula:**
  $$\text{Deduct Value (DV)} = (N_{\text{digs}} \times 12) + (N_{\text{patches}} \times 8) + (N_{\text{cracks}} \times 4) + (N_{\text{potholes}} \times 15)$$
  $$\text{PCI} = \max\left(0.0, \min\left(100.0, 100.0 - \text{DV}\right)\right)$$
* **State Vector Generation:** Output state array $S = [\text{PCI}, N_{\text{digs}}, N_{\text{patches}}, N_{\text{cracks}}, N_{\text{potholes}}, \text{Road Age}]$.

### Module 3: Reinforcement Learning Maintenance Engine (`/rl_engine`)
* **Framework:** Gymnasium (`gymnasium`), Stable-Baselines3 (DQN / PPO).
* **Observation Space:** Continuous 6-value array bounded by `Box(low=[0, 0, 0, 0, 0, 0], high=[100, 50, 50, 50, 50, 30])`.
* **Action Space:** `Discrete(3)`
  * `0`: **Do Nothing** (Cost: 0, degradation continues based on traffic and existing defects).
  * `1`: **Preventive Maintenance** (Cost: 15, repairs open digs/patches, restores +20 PCI).
  * `2`: **Major Resurfacing** (Cost: 60, resets defects to 0, restores PCI to 100).
* **Reward Function Specification:**
  $$R = \left(\frac{\text{PCI}}{10.0}\right) - C_{\text{action}} - P_{\text{failure}}$$
  Where $C_{\text{action}} \in \{0, 15, 60\}$ and penalty $P_{\text{failure}} = 50$ if $\text{PCI} < 40$.

---

## 5. API Specification (FastAPI)

### `POST /api/v1/analyze-video`
* **Request:** `multipart/form-data` containing an `.mp4` or `.avi` video file.
* **Response Body (`application/json`):**
```json
{
  "status": "success",
  "filename": "inspection_run_01.mp4",
  "total_segments_analyzed": 5,
  "summary": {
    "overall_pci": 64.5,
    "recommended_action": "Preventive Repair Required",
    "total_defects": {
      "digs": 3,
      "patches": 7,
      "cracks": 12,
      "potholes": 1
    }
  },
  "segment_details": [
    {
      "segment_id": 1,
      "pci_score": 52.0,
      "defects": {"digs": 2, "patches": 3, "cracks": 4, "potholes": 1},
      "rl_recommended_action": 1,
      "action_name": "Preventive Repair"
    }
  ]
}
```

---

## 6. Dashboard & User Interface Specifications
* **Tool:** Streamlit frontend (`/dashboard/app.py`).
* **Components:**
  1. **Upload Panel:** Drag-and-drop support for `.mp4` dashcam recordings.
  2. **Video Player & Real-time Canvas:** OpenCV output stream rendering bounding boxes over detected digs and defects.
  3. **PCI Metric Gauge:** Visual gauge displaying overall road segment condition score.
  4. **Policy Recommendation Box:** Dynamic notification banner displaying RL prediction (Green: Optimal, Yellow: Scheduled Repair, Red: Critical Reconstruction).
  5. **Export Section:** Button to download inspection logs as a JSON or CSV file.

---

## 7. Success Metrics & Validation Strategy
* **Computer Vision Precision/Recall:** mAP@0.5 $\ge 0.82$ on road defect evaluation sets.
* **RL Policy Convergence:** Policy must achieve stable positive rewards over 50,000 training timesteps without dropping PCI below critical threshold (40).
* **Inference Speed:** Target $\ge 24\text{ FPS}$ on GPU instances and $\ge 10\text{ FPS}$ on standard CPU hardware.