@echo off
title AI Smart Road Platform - Mobile Cloud Server
echo =====================================================================
echo  AI SMART ROAD DEFECT DETECTION ^& PREDICTIVE MAINTENANCE PLATFORM
echo =====================================================================
echo.
echo [1/2] Starting Streamlit Server...
start "Streamlit Server" cmd /k "streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501"
timeout /t 4 /nobreak >nul
echo.
echo [2/2] Launching Public HTTPS Tunnel for Mobile Access...
echo Open the trycloudflare.com URL below on your smartphone browser.
echo =====================================================================
npx -y cloudflared tunnel --url http://localhost:8501
pause
