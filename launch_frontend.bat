@echo off
title RoadDetector AI - Full-Stack Modern Platform
echo =====================================================================
echo  ROADDETECTOR AI - HIGH-FIDELITY WEB PLATFORM
echo =====================================================================
echo.
echo [1/2] Starting FastAPI Backend on port 8000...
start "RoadDetector API Backend" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
echo.
echo [2/2] Launching React + TypeScript + Tailwind UI on port 3000...
cd frontend
start "RoadDetector Modern UI" cmd /k "npm run dev -- --host 0.0.0.0 --port 3000"
timeout /t 3 /nobreak >nul
echo.
echo =====================================================================
echo  Platform is now running!
echo  Browser URL: http://localhost:3000
echo  Backend Swagger Docs: http://localhost:8000/docs
echo =====================================================================
start http://localhost:3000
pause
