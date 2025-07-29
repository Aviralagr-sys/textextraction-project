@echo off
echo Starting Flask server...
start "Flask Server" cmd /k "cd /d %~dp0 && python app.py"

REM Wait 4 seconds for Flask backend to start properly
timeout /t 4 > nul

echo Starting Streamlit UI...
start "Streamlit UI" cmd /k "cd /d %~dp0 && streamlit run ui.py"

REM Do NOT open browser explicitly here to avoid double openings
REM Simply access Streamlit at http://localhost:8501 (or whichever port Streamlit reports)

exit
