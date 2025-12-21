@echo off
REM IndexTTS API Start Script for Windows

echo.
echo Starting IndexTTS API Server...
echo.

REM Check if running from project root
if not exist "api\main.py" (
    echo Error: Please run this script from the project root directory
    echo    Example: start_api.bat
    pause
    exit /b 1
)

REM Check if checkpoints directory exists
if not exist "checkpoints" (
    echo Warning: checkpoints\ directory not found
    echo    The model will need to be loaded manually or download the checkpoints
    echo.
)

REM Check if API dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Warning: FastAPI not found. Installing API dependencies...
    pip install -r api\requirements.txt
)

echo.
echo Starting server on http://localhost:8000
echo    - API docs: http://localhost:8000/docs
echo    - WebUI: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload