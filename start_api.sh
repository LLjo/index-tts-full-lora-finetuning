#!/bin/bash
# IndexTTS API Start Script

# System CUDA 13.1 LD_LIBRARY_PATH conflicts with this project's torch (cu128
# wheels ship their own CUDA libs). Drop it for the duration of this script.
unset LD_LIBRARY_PATH

echo "🚀 Starting IndexTTS API Server..."
echo ""

# Check if running from project root
if [ ! -f "api/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Example: ./start_api.sh"
    exit 1
fi

# Check if checkpoints directory exists
if [ ! -d "checkpoints" ]; then
    echo "⚠️  Warning: checkpoints/ directory not found"
    echo "   The model will need to be loaded manually or download the checkpoints"
fi

# Check if API dependencies are installed (in the project venv via uv).
uv run --no-sync python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: FastAPI not found. Run ./install.sh to set up the venv."
    exit 1
fi

# Start the server
echo ""
echo "✅ Starting server on http://localhost:8000"
echo "   - API docs: http://localhost:8000/docs"
echo "   - WebUI: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run --no-sync python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload