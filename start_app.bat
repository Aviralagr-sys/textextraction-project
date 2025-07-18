@echo off
echo Starting Flask server...
start cmd /k "cd /d %~dp0 && python app.py"

timeout /t 2 > nul

echo Starting Streamlit UI...
start "" "C:\Users\aviral\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\streamlit.cmd" run ui.py

exit
