#!/usr/bin/env bash
# IndexTTS install script.
#
# This project is fussy about its environment for three reasons:
#   1. It pins torch built for CUDA 12.8 (cu128). Any LD_LIBRARY_PATH that
#      points at a *different* system CUDA (e.g. 13.x) shadows torch's bundled
#      cuDNN at runtime and breaks `import torch`.
#   2. flash-attn cannot be compiled from source unless the system nvcc matches
#      torch's CUDA major version. We sidestep this with a prebuilt wheel.
#   3. Triton / DeepSpeed / BigVGAN custom kernels JIT-compile native code at
#      first inference and need the Python 3.10 development headers
#      (`/usr/include/python3.10/Python.h`).
#
# This script enforces those preconditions before installing anything.
set -euo pipefail

# ─── (1) Drop system CUDA from LD_LIBRARY_PATH for this script ──────────────
unset LD_LIBRARY_PATH

# ─── (2) Verify Python 3.10 development headers ─────────────────────────────
if [ ! -f /usr/include/python3.10/Python.h ]; then
    echo "❌ Missing: /usr/include/python3.10/Python.h"
    echo
    echo "   Triton, DeepSpeed, and torch.compile JIT-compile C/CUDA helpers at"
    echo "   first inference. They need the Python 3.10 development headers."
    echo
    echo "   Install with:"
    echo "       sudo apt install python3.10-dev"
    echo
    echo "   Then re-run this script."
    exit 1
fi

# ─── (3) Warn if system CUDA major doesn't match torch's (cu128 → CUDA 12) ──
if command -v nvcc >/dev/null 2>&1; then
    nvcc_major=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "?")
    if [ "$nvcc_major" != "12" ]; then
        echo "⚠️  System nvcc reports CUDA $nvcc_major, but this project ships"
        echo "    torch built for CUDA 12.8. Custom CUDA kernels that JIT-compile"
        echo "    (BigVGAN's anti-alias, etc.) will fail and fall back to torch."
        echo "    The app still works; you just lose a small bit of acceleration."
        echo
    fi
fi

# ─── (4) Sync everything except flash-attn ──────────────────────────────────
# flash-attn 2.8.3 has no CUDA-13 wheels and source builds need CUDA 12 nvcc.
# We install the prebuilt cu12 wheel below instead.
uv sync --all-extras --no-install-package flash-attn

# ─── (5) Install prebuilt flash-attn wheel matching cp310/torch2.8/cu12 ─────
# Wheel naming: see https://github.com/Dao-AILab/flash-attention/releases
# We pick cxx11abiTRUE because torch 2.8 on Linux is built with the new ABI.
uv pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

# ─── (6) Verify the venv actually works end-to-end ──────────────────────────
uv run --no-sync python -c "import torch, flash_attn; assert torch.cuda.is_available(), 'CUDA not available'; print('✅ torch', torch.__version__, '+ flash-attn', flash_attn.__version__, '+ CUDA ready')"

# ─── (7) Download the IndexTTS-2 checkpoints (~10 GB) ───────────────────────
uv run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints

echo
echo "✅ Install complete."
echo
echo "Start the API:  bash start_api.sh"
echo "Start the WebUI: ./run uv run python webui.py"
echo
echo "Note: if your shell has LD_LIBRARY_PATH pointing at a non-CUDA-12 toolkit"
echo "      (e.g. /usr/local/cuda-13.x/lib64), use the ./run wrapper for any"
echo "      ad-hoc 'uv run' commands so the env is scrubbed before Python starts."
