#!/bin/bash
# IndexTTS + Home Assistant (Wyoming) orchestrator launcher.
# See scripts/serve_ha.py for what this brings up.

# System CUDA 13.1 LD_LIBRARY_PATH conflicts with this project's torch (cu128
# wheels ship their own CUDA libs). Drop it for the duration of this script —
# subprocesses spawned by serve_ha.py inherit this clean env.
unset LD_LIBRARY_PATH

# Sanity-check that the project venv is set up.
uv run --no-sync python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Project venv not ready. Run ./install.sh first."
    exit 1
fi

exec uv run --no-sync python scripts/serve_ha.py "$@"
