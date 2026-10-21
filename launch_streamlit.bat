@echo off
title AI Smart Road Platform - Streamlit Web Dashboard
echo =====================================================================
echo  AI SMART ROAD DEFECT DETECTION ^& PREDICTIVE MAINTENANCE PLATFORM
echo  Streamlit Interactive Web Dashboard
echo =====================================================================
echo.
echo [1/2] Initializing Streamlit application on http://localhost:8501...
start "RoadDetector Streamlit App" cmd /k "streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"
timeout /t 3 /nobreak >nul
echo.
echo [2/2] Opening browser to http://localhost:8501...
start http://localhost:8501
echo.
echo =====================================================================
echo  Streamlit is now running!
echo  Local URL:   http://localhost:8501
echo  Network URL: Check command prompt window
echo =====================================================================
pause
