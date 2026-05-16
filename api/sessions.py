"""Per-session mel-token prefix for cross-call prosody continuity.

Mechanism: capture the last N mel codes from each call's GPT output, store
them under an opaque `session_id`, and inject them as the GPT's `input_tokens`
prefix on the next call. The GPT continues autoregressively from those codes
— pitch contour, tempo, and prosodic momentum carry through — while the
`spk_audio_prompt` stays as the **original** speaker reference, so the
`style` / `spk_cond_emb` / `ref_mel` extraction sees only clean audio. No
synthesized audio ever feeds back into the conditioning path, so there's no
photocopy-of-a-photocopy drift.

This replaces the earlier audio-tail-as-spk_prompt mechanism (see the
revision note in docs/SESSION_CONTINUITY_PLAN.md). Strictly token-domain.

Caller contract:
  * Get the previous prefix with `get_session_prefix(session_id)`.
  * Pass it through `streaming_inference_v2(prefix_codes=...)` (or the
    pattern wrapper). When prefix codes are present, streaming_v2 forces
    the non-accel HF generate path — the accel engine's CUDA graphs are
    captured at a fixed prefix length and would have to recapture on
    every call.
  * Receive the full mel-code tensor via the `on_codes_complete` callback
    and persist the tail with `store_session_prefix(session_id, codes)`.

Sessions are in-process only — they are conversational state, not durable.
TTL-swept on a background task; LRU-capped at SESSION_MAX_COUNT.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

LOG = logging.getLogger("indextts.sessions")


# ────────────────────────────── configuration ──────────────────────────────


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _envb(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SESSION_ENABLED: bool = _envb("INDEXTTS_SESSION_ENABLED", True)
SESSION_TTL_S: float = _envf("INDEXTTS_SESSION_TTL_S", 90.0)
SESSION_MAX_COUNT: int = _envi("INDEXTTS_SESSION_MAX_COUNT", 64)

# User-facing prefix length in seconds. The API converts this to a mel-token
# count using a ~50 tokens/s heuristic. 4.0 s (~200 tokens) fully covers
# the codes for typical HA-style replies so we can hand the model exact
# (codes, text) alignment instead of an approximate clip.
SESSION_DEFAULT_TAIL_S: float = _envf("INDEXTTS_SESSION_DEFAULT_TAIL_S", 4.0)
SESSION_TAIL_MIN_S: float = 0.3
SESSION_TAIL_MAX_S: float = 8.0

# Hard cap on how many mel tokens we retain as the GPT prefix. ~50 mel
# tokens ≈ 1 second of audio for IndexTTS2's mel rate. 200 tokens (~4 s)
# is enough to fully cover the codes for most short HA-style replies, so
# the model gets EXACT (codes, text) alignment without needing the
# clipping heuristic. Bigger = stronger continuation; capped at 400 so
# the prefix can't eat the full max_mel_tokens budget on the next call.
SESSION_PREFIX_TOKENS: int = _envi("INDEXTTS_SESSION_PREFIX_TOKENS", 200)
SESSION_PREFIX_TOKENS_MAX: int = 400
# Below this many tokens, the prefix is too small to be useful — drop it
# rather than feed a 5-token nub the GPT can't extract anything from.
SESSION_PREFIX_TOKENS_MIN: int = 12


# ────────────────────────────── data model ──────────────────────────────


@dataclass
class TTSSession:
    """One conversational session: the trailing mel-code prefix from the
    previous call AND the text fragment those codes correspond to.

    The text is critical — without it, the GPT misinterprets the codes as
    "I've already generated mel for the start of THIS text" and skips the
    first words of the new call. With it, the GPT sees aligned state:
    "I produced codes for X, now produce codes for Y." See `store_session_prefix`
    for the word/code clipping heuristic that keeps both bounded."""

    session_id: str
    prefix_codes: torch.Tensor  # 1-D long, CPU
    prefix_text: str            # text the codes correspond to (clipped tail)
    last_used_ns: int


_sessions: Dict[str, TTSSession] = {}
_sessions_lock = threading.Lock()
_sweeper_task: Optional[asyncio.Task] = None


def _now_ns() -> int:
    return time.monotonic_ns()


def _coerce_prefix_codes(codes: torch.Tensor, max_tokens: int) -> Optional[torch.Tensor]:
    """Reduce a generated-codes tensor to the trailing slice we want to keep.

    Returns CPU long 1-D tensor of length ≤ max_tokens, or None if there
    aren't enough tokens to be useful."""
    if codes is None:
        return None
    if codes.dim() == 2:
        # (batch, seq) — streaming generates batch=1, just take row 0.
        codes = codes[0]
    if codes.dim() != 1:
        return None
    if codes.numel() < SESSION_PREFIX_TOKENS_MIN:
        return None
    n = min(int(max_tokens), int(codes.numel()))
    if n < SESSION_PREFIX_TOKENS_MIN:
        return None
    return codes[-n:].to(dtype=torch.long, device="cpu").contiguous()


# ────────────────────────────── store API ──────────────────────────────


# Codes-per-word heuristics for IndexTTS2. The model's actual rate varies
# (~10-16 codes/word depending on the word and prosody). We use TWO ratios:
#
#   _MIN_RATE  — generous "are the codes definitely enough to cover the
#                whole text?" check. If codes ≥ words × _MIN_RATE, the
#                codes provably cover all of prev_text and we keep it
#                verbatim — exact alignment, no guessing.
#   _TRIM_RATE — conservative "if we MUST trim, how many words can we
#                keep?". Higher = fewer words kept = codes over-cover the
#                kept text. This prevents the model from feeling like it
#                has unfinished business in the prefix (which manifests as
#                ECHOING the tail of prev_text at the start of the new
#                call's audio — the actual bug this whole helper exists
#                to avoid).
_MEL_CODES_PER_WORD_MIN = 11
_MEL_CODES_PER_WORD_TRIM = 18


def _clip_text_to_match_codes(text: str, n_codes: int) -> str:
    """Pick the trailing slice of `text` that the stored codes are sure to
    cover. When codes are abundant we keep the whole text (exact
    alignment); when they're scarce we keep FEWER words than the average
    rate would suggest, so the codes always over-cover rather than
    under-cover the kept text.

    Under-coverage = "I haven't finished saying this yet" → model echoes
    the tail. Over-coverage = "I'm slightly past this" → model moves
    cleanly into the new text. Over is fine; under is the audible bug."""
    words = text.strip().split()
    if not words:
        return ""
    # Fast path: codes definitely cover all the words → use them all.
    if n_codes >= len(words) * _MEL_CODES_PER_WORD_MIN:
        return " ".join(words)
    # Trim conservatively. n_codes // 18 keeps fewer words than the codes
    # actually cover, guaranteeing the model sees a "finished" prefix.
    n_words = max(1, n_codes // _MEL_CODES_PER_WORD_TRIM)
    return " ".join(words[-n_words:])


def store_session_prefix(
    session_id: str,
    codes: torch.Tensor,
    text: str,
    max_tokens: Optional[int] = None,
) -> bool:
    """Persist (codes, text) pair under session_id. `text` is the request
    text that produced `codes`; both are clipped to ~max_tokens worth so
    the GPT's autoregressive prefix budget stays bounded on the next call.

    Returns True on success, False if codes are too short/malformed."""
    if not SESSION_ENABLED or not session_id:
        return False
    cap = max_tokens if max_tokens is not None else SESSION_PREFIX_TOKENS
    cap = max(SESSION_PREFIX_TOKENS_MIN, min(SESSION_PREFIX_TOKENS_MAX, int(cap)))
    trimmed = _coerce_prefix_codes(codes, cap)
    if trimmed is None:
        LOG.debug("session %s: prefix codes rejected (too short or malformed)", session_id)
        return False
    text_clip = _clip_text_to_match_codes(text or "", trimmed.numel())
    if not text_clip:
        # No text means we can't align the codes — without text, injecting the
        # codes would re-introduce the "first words skipped" bug.
        LOG.debug("session %s: rejected — no text to align codes to", session_id)
        return False
    with _sessions_lock:
        _sessions[session_id] = TTSSession(
            session_id=session_id,
            prefix_codes=trimmed,
            prefix_text=text_clip,
            last_used_ns=_now_ns(),
        )
        if len(_sessions) > SESSION_MAX_COUNT:
            oldest = min(_sessions.values(), key=lambda s: s.last_used_ns)
            _sessions.pop(oldest.session_id, None)
    LOG.debug(
        "session %s: stored prefix (%d codes, text=%r)",
        session_id, trimmed.numel(), text_clip[:60],
    )
    return True


def get_session_prefix(session_id: str) -> Optional["TTSSession"]:
    """Return the stored session (codes + text) or None. Also refreshes
    last_used_ns so the sweeper doesn't drop it mid-utterance."""
    if not SESSION_ENABLED or not session_id:
        return None
    with _sessions_lock:
        s = _sessions.get(session_id)
        if s is None:
            return None
        s.last_used_ns = _now_ns()
        return s


def delete_session(session_id: str) -> bool:
    if not session_id:
        return False
    with _sessions_lock:
        return _sessions.pop(session_id, None) is not None


def list_sessions() -> List[Dict[str, float]]:
    now = _now_ns()
    out: List[Dict[str, float]] = []
    with _sessions_lock:
        for s in _sessions.values():
            out.append({
                "session_id": s.session_id,
                "age_s": (now - s.last_used_ns) / 1e9,
                "prefix_tokens": int(s.prefix_codes.numel()),
                "prefix_text": s.prefix_text,
            })
    return out


# ────────────────────────────── sweeper ──────────────────────────────


async def _sweep_loop(interval_s: float = 30.0) -> None:
    ttl_ns = int(SESSION_TTL_S * 1e9)
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        now = _now_ns()
        dropped = 0
        with _sessions_lock:
            stale = [sid for sid, s in _sessions.items() if now - s.last_used_ns > ttl_ns]
            for sid in stale:
                _sessions.pop(sid, None)
                dropped += 1
        if dropped:
            LOG.info("session sweeper dropped %d idle sessions", dropped)


def start_sweeper() -> None:
    """Idempotent — safe to call from FastAPI startup."""
    global _sweeper_task
    if _sweeper_task is not None and not _sweeper_task.done():
        return
    _sweeper_task = asyncio.create_task(_sweep_loop())


def stop_sweeper() -> None:
    global _sweeper_task
    if _sweeper_task is not None:
        _sweeper_task.cancel()
        _sweeper_task = None
