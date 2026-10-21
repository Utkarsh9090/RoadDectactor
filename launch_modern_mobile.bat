@echo off
title RoadDetector AI - Mobile Cloud HTTPS Platform
echo =====================================================================
echo  ROADDETECTOR AI - MOBILE ^& CLOUD HTTPS ACCESS
echo =====================================================================
echo.
echo [1/3] Starting FastAPI Backend on port 8000...
start "RoadDetector API Backend" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
echo.
echo [2/3] Launching React Modern Frontend on port 3000...
cd frontend
start "RoadDetector Modern UI" cmd /k "npm run dev -- --host 0.0.0.0 --port 3000"
cd ..
timeout /t 4 /nobreak >nul
echo.
echo [3/3] Launching Public HTTPS Cloudflare Tunnel...
echo Open the trycloudflare.com URL below on any smartphone or tablet.
echo =====================================================================
npx -y cloudflared tunnel --url http://localhost:3000
pause
