# 🛣️ Streamlit Deployment Guide: AI Smart Road Defect Platform

This guide outlines how to run and deploy the **RoadDetector AI** interactive Streamlit dashboard both locally and to public cloud environments (Streamlit Community Cloud and Instant HTTPS Tunnels).

---

## 1. Quick Local Launch

To run the Streamlit dashboard immediately on your computer:

### Option A: One-Click Batch Script
Double-click:
```powershell
.\launch_streamlit.bat
```
This launches Streamlit on `http://localhost:8501` and opens it in your default web browser.

### Option B: Command Line
```powershell
streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```
Then navigate to:
- **Local Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Local Network / Phone on same Wi-Fi:** `http://<your-ip-address>:8501`

---

## 2. Instant Public HTTPS URL (Cloudflare Tunnel)

If you want an immediate live HTTPS URL accessible from smartphones or remote users worldwide without setting up cloud accounts:

1. In Terminal 1, run Streamlit:
   ```powershell
   streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
   ```
2. In Terminal 2, run Cloudflare Tunnel:
   ```powershell
   npx -y cloudflared tunnel --url http://localhost:8501
   ```
3. Cloudflare will print a secure URL like:
   ```
   https://random-subdomain.trycloudflare.com
   ```
   Share this URL with anyone for instant live access.

---

## 3. Permanent Cloud Deployment on Streamlit Community Cloud (share.streamlit.io)

Streamlit Community Cloud provides 100% free hosting directly integrated with your GitHub repository.

### Prerequisites Configured in Repository:
- ✅ **Configuration file:** [`.streamlit/config.toml`](file:///d:/Workspace/Projects/RoadDetector/.streamlit/config.toml) (Sets dark theme, ports, 250MB upload limits)
- ✅ **System dependencies:** [`packages.txt`](file:///d:/Workspace/Projects/RoadDetector/packages.txt) (Installs `ffmpeg`, `libgl1`, and OpenGL libraries required for OpenCV)
- ✅ **Python packages:** [`requirements.txt`](file:///d:/Workspace/Projects/RoadDetector/requirements.txt)
- ✅ **Trained Policy:** [`data/models/dqn_road_policy.zip`](file:///d:/Workspace/Projects/RoadDetector/data/models/dqn_road_policy.zip)
- ✅ **Real Video Datasets & Pre-computed Cache:** [`data/real_video_dataset.json`](file:///d:/Workspace/Projects/RoadDetector/data/real_video_dataset.json)

### Step-by-Step Deployment Instructions:

1. **Commit & Push to GitHub:**
   ```powershell
   git add .
   git commit -m "Configure Streamlit deployment with real video datasets and trained DQN policy"
   git push origin main
   ```

2. **Log in to Streamlit Community Cloud:**
   - Go to [https://share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account.

3. **Deploy the App:**
   - Click the blue **"Create app"** button.
   - Select your repository (`Utkarsh9090/RoadDectactor` or your repository name).
   - Set **Branch:** `main` (or `master`).
   - Set **Main file path:** `dashboard/app.py`
   - *(Optional)* Choose a custom App URL (e.g., `road-defect-ai.streamlit.app`).
   - Click **Deploy!**

4. **Monitoring & Verification:**
   - Streamlit Cloud will automatically install packages from `packages.txt` and `requirements.txt`.
   - Within 2-3 minutes, your application will be live at `https://<your-app-name>.streamlit.app`!

---

## 4. Key Streamlit Features Included

- **🎬 Real Highway & Urban Videos Catalog:** Select any of the 3 real videos directly from a dropdown with instant playback streaming.
- **⚡ Pre-computed AI Cache:** Toggle instant report generation from the real video evaluations without waiting for frame processing.
- **📁 Custom Video Upload:** Supports uploading `.mp4`, `.avi`, `.mov`, `.mkv` files up to 250MB.
- **📹 Live Camera Recording & Snapshot:** Record short dashcam video segments or capture single-frame road snapshots directly in the browser.
- **📊 ASTM D6433 Pavement Condition Index:** Interactive Plotly gauge with color-coded condition tiers (Good, Satisfactory, Fair, Poor, Very Poor, Serious, Failed).
- **🤖 Deep Q-Network Policy Directives:** Displays automated maintenance interventions (Do Nothing, Preventive Repair, Major Resurfacing) and 30-day scheduling guidance.
- **📥 One-Click Export:** Download full inspection reports as **JSON** and segment logs as **CSV**.
