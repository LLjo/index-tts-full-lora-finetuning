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


_env_primary = ROOT / ".env.indextts"
if _env_primary.exists():
    _load_env_file(_env_primary)
else:
    print(f"[serve_ha] WARNING: {_env_primary.name} not found — running with "
          f"defaults only. Run `cp .env.indextts.example .env.indextts` and "
          f"edit it to customize preset, solver overrides, default speaker, etc.",
          flush=True)

API_HOST = os.environ.get("INDEXTTS_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("INDEXTTS_API_PORT", "8000"))
WY_PORT = int(os.environ.get("INDEXTTS_WYOMING_PORT", "10200"))
DEFAULT_SPEAKER = os.environ.get("INDEXTTS_DEFAULT_SPEAKER") or None
HEALTH_TIMEOUT = int(os.environ.get("INDEXTTS_HEALTH_TIMEOUT_S", "180"))
LOAD_TIMEOUT = int(os.environ.get("INDEXTTS_LOAD_TIMEOUT_S", "600"))


def log(msg: str) -> None:
    print(f"[serve_ha] {msg}", flush=True)


def http_post(url: str, timeout: int = 600, body: dict | None = None) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body_text = r.read().decode("utf-8", "replace")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError:
            return {"raw": body_text}


def _per_speaker_student_path(speaker: str) -> Path:
    """Where the distilled CFM student lives on disk (if it was trained for
    this speaker). The API's /distill/activate endpoint copies this file into
    checkpoints/s2mel_distilled.pth + writes a sidecar so the bridge can map
    the active student back to its source speaker."""
    return ROOT / "training" / speaker / "cfm_reflow_student" / "best.pth"


def maybe_activate_distilled(api_port: int, speaker: str) -> bool:
    """If `speaker` has a distilled CFM student on disk, swap it into the
    active slot via /distill/activate. Returns True iff activation happened
    (caller should reload the base model). Returns False (and logs the
    reason) for: no student on disk, API error, or non-fatal 4xx. Never
    raises — distillation is an optimization, not a hard requirement."""
    student = _per_speaker_student_path(speaker)
    if not student.exists():
        log(f"no distilled student at {student.relative_to(ROOT)} — "
            f"falling back to the global s2mel_distilled.pth if present.")
        return False
    log(f"activating distilled CFM student for speaker={speaker} "
        f"(source: {student.relative_to(ROOT)})…")
    try:
        out = http_post(f"http://localhost:{api_port}/distill/activate",
                        timeout=30, body={"speaker": speaker})
        log(f"distill activate: {out.get('status') or out}")
        return True
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:
            detail = e.reason
        log(f"distill activate FAILED ({e.code}): {detail} — continuing "
            f"without per-speaker student.")
        return False
    except Exception as e:
        log(f"distill activate FAILED ({e}) — continuing without per-speaker student.")
        return False


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

    # If the default speaker has a distilled CFM student on disk, copy it
    # into the active slot BEFORE loading the base model — the model reads
    # checkpoints/s2mel_distilled.pth at load time, so activating after
    # would require a costly reload.
    if DEFAULT_SPEAKER:
        maybe_activate_distilled(API_PORT, DEFAULT_SPEAKER)

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
