#!/usr/bin/env python3
"""Orchestrator: bring up the IndexTTS API + Wyoming bridge for Home Assistant.

Run this once when you want HA to be able to talk to IndexTTS. It will:
  1. Start the FastAPI server (uvicorn, no --reload).
  2. Wait for /health to report healthy.
  3. POST /models/load/base to bring the model up (loads any active distilled
     student automatically — that's existing behavior of IndexTTS2.__init__).
  4. If INDEXTTS_DEFAULT_SPEAKER is set, POST /models/load/<speaker> to merge
     that speaker's character LoRA into the GPT.
  5. Start tools/wyoming_indextts.py as a child process listening on
     INDEXTTS_WYOMING_PORT (default 10200).
  6. Watch both. SIGINT/SIGTERM terminate both cleanly.

Loads env from .env.indextts at the project root if present. Doesn't require
any extra deps beyond python-stdlib + uv-installed packages.

Usage:
    python scripts/serve_ha.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Bare-bones .env loader (no python-dotenv dep)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes if present
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        # Don't overwrite already-exported env vars (so CLI overrides win)
        os.environ.setdefault(key, val)


_load_env_file(ROOT / ".env.indextts")

API_HOST = os.environ.get("INDEXTTS_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("INDEXTTS_API_PORT", "8000"))
WY_PORT = int(os.environ.get("INDEXTTS_WYOMING_PORT", "10200"))
DEFAULT_SPEAKER = os.environ.get("INDEXTTS_DEFAULT_SPEAKER") or None
HEALTH_TIMEOUT = int(os.environ.get("INDEXTTS_HEALTH_TIMEOUT_S", "180"))
LOAD_TIMEOUT = int(os.environ.get("INDEXTTS_LOAD_TIMEOUT_S", "600"))


def log(msg: str) -> None:
    print(f"[serve_ha] {msg}", flush=True)


def http_post(url: str, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}


def wait_for_health(url: str, timeout_s: int) -> bool:
    """Poll /health until it reports healthy, or timeout."""
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                data = json.loads(r.read())
                if data.get("status") == "healthy":
                    return True
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(1)
    log(f"health never came up after {timeout_s}s (last error: {last_err})")
    return False


_PROCS: list[subprocess.Popen] = []


def _cleanup(*_):
    log("shutting down child processes…")
    for p in _PROCS:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    # Give them 5s to wind down, then SIGKILL the holdouts
    deadline = time.time() + 5
    for p in _PROCS:
        try:
            remaining = max(0.5, deadline - time.time())
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
    sys.exit(0)


def main() -> int:
    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    venv_python = sys.executable

    log(f"starting uvicorn on http://{API_HOST}:{API_PORT}")
    api_proc = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "api.main:app",
         "--host", API_HOST, "--port", str(API_PORT)],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    _PROCS.append(api_proc)

    # If uvicorn dies before we can ping it, surface the exit fast
    health_url = f"http://localhost:{API_PORT}/health"
    log(f"waiting for {health_url} (timeout {HEALTH_TIMEOUT}s)…")
    if not wait_for_health(health_url, HEALTH_TIMEOUT):
        _cleanup()
        return 1

    log("API is healthy — loading base model (30-60s)…")
    try:
        out = http_post(f"http://localhost:{API_PORT}/models/load/base", timeout=LOAD_TIMEOUT)
        log(f"base loaded: {out.get('message') or out}")
    except Exception as e:
        log(f"base load FAILED: {e}")
        _cleanup()
        return 1

    if DEFAULT_SPEAKER:
        log(f"loading default speaker LoRA: {DEFAULT_SPEAKER}")
        try:
            out = http_post(f"http://localhost:{API_PORT}/models/load/{DEFAULT_SPEAKER}",
                            timeout=LOAD_TIMEOUT)
            log(f"speaker loaded: {out.get('message') or out}")
        except Exception as e:
            log(f"speaker load failed (will lazy-load on first HA request): {e}")

    # Pass-through env for the bridge (it reads INDEXTTS_*)
    bridge_env = os.environ.copy()
    bridge_env.setdefault("INDEXTTS_API_URL", f"http://localhost:{API_PORT}")
    bridge_env.setdefault("INDEXTTS_WYOMING_PORT", str(WY_PORT))

    log(f"starting Wyoming bridge on tcp://0.0.0.0:{WY_PORT}")
    bridge_proc = subprocess.Popen(
        [venv_python, str(ROOT / "tools" / "wyoming_indextts.py")],
        cwd=ROOT,
        env=bridge_env,
    )
    _PROCS.append(bridge_proc)

    log("✅ all services up. Ctrl+C to stop.")
    # Watch loop — if either child dies, take both down
    try:
        while True:
            for p in _PROCS:
                rc = p.poll()
                if rc is not None:
                    log(f"child pid={p.pid} exited rc={rc} — stopping")
                    _cleanup()
                    return rc
            time.sleep(1)
    except KeyboardInterrupt:
        _cleanup()
        return 0


if __name__ == "__main__":
    sys.exit(main())
