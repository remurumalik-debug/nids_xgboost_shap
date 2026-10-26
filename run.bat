@echo off
title Explainable NIDS - Flask Backend

rem Ensure virtualenv in project root is activated when possible
pushd "%~dp0\.."
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate" (
    call ".venv\Scripts\activate"
) else (
    echo No local virtualenv found at .venv — continuing with system Python
)
popd

cd /d "%~dp0\flask_backend"

if not exist "model\xgb_model.json" (
    echo Model not found. Training...
    python train.py
    if errorlevel 1 (
        echo Training failed!
        pause
        exit /b 1
    )
)

echo Starting Flask server on http://localhost:5050
echo Press CTRL+C to stop
python app.py
pause