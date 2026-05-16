# IndexTTS2 — Streaming + LoRA + Home Assistant

Fork of IndexTTS2 focused on three things: (1) sub-200 ms time-to-first-audio
streaming, (2) per-speaker character LoRAs that capture *how* someone talks
(stutters, fillers, pauses), and (3) a Wyoming bridge that lets Home Assistant
use any trained voice as its TTS engine.

The full picture for the streaming + distillation work lives in
`docs/STREAMING_LATENCY_RESULTS.md`. The HA integration is documented in
`docs/HOMEASSISTANT_INTEGRATION_PLAN.md`. Read those before making non-trivial
changes — they encode design decisions that are not derivable from the code.

---

## Running it

```bash
bash start_api.sh        # FastAPI on :8000 (WebUI at /, API docs at /docs)
bash start_ha.sh         # orchestrator: API + Wyoming bridge on :10200 for HA
```

Both scripts `unset LD_LIBRARY_PATH` because the system CUDA 13.1 libs conflict
with the cu128 wheels torch ships with. Both run inside the project venv via
`uv run --no-sync`. `start_api.sh` uses `--reload`; `start_ha.sh` does not (it
runs `scripts/serve_ha.py`, which lazy-loads the base model and the default
speaker LoRA, then `exec`s `tools/wyoming_indextts.py`).

Production/HA defaults live in `.env.indextts`:
`INDEXTTS_DEFAULT_SPEAKER`, `INDEXTTS_STREAMING_PRESET` (typically
`fast_quality` or `balanced_distilled`), and the solver overrides
(`single_step` / 1 step / CFG 0.0) used when the distilled student is active.

---

## What the recent commits actually did

`8b7eaf6 — distillation` is the most important one to understand. It built the
full CFM reflow-distillation pipeline (snapshot teacher → JSONL manifest →
paired data → train student → A/B → activate), added `solve_single_step()` to
`indextts/s2mel/modules/flow_matching.py`, and rewrote `indextts/streaming_v2.py`
around a synth-worker thread, a GPT/synth serialization gate, a speaker-cond
cache, and presets including `ultra_fast_distilled` and `balanced_distilled`.
After this commit, first-chunk audio with a distilled student drops to ~200 ms.

`4d5b107 — remove old training, new train method` deleted the legacy
verbatim/pattern training UI and added the **character LoRA** trainer
(`tools/train_character_lora.py` + `tools/prepare_character_dataset.py`). The
new trainer's core trick: the loss is weighted **per mel-token** using a
boolean `stutter_mask`, so stutters/fillers actually shape the gradient
instead of being washed out by surrounding clean tokens. Input text stays
clean — the LoRA learns the speaker's delivery, not their disfluent transcript.

`8271da2 → 376fc98` (4 commits) — Wyoming bridge for HA: replaces
`wyoming-piper`, advertises one Wyoming voice per loaded character LoRA, and
streams sentence-by-sentence to keep HA responsive on long replies.

These commits collectively define the project's current shape. **Older
training paths (verbatim trainer, pattern embeddings as the primary mechanism)
are still in the repo but are not the current production path.** Don't add
features that assume they are.

---

## Layout

| Path | What's there |
| --- | --- |
| `api/main.py` | FastAPI app — inference, training, distillation, model management endpoints. Single big file. |
| `api/static/` | WebUI (vanilla JS, no build step). Tabs: Inference, Training, Distillation, Test Lab, Speakers. |
| `indextts/infer_v2.py` | `IndexTTS2` model class. Loads base + optional distilled student via `s2mel_distilled_checkpoint`. |
| `indextts/streaming_v2.py` | The streaming pipeline. `streaming_inference_v2()` + the preset factories used everywhere else. |
| `indextts/accel/accel_engine.py` | CUDA-graph GPT decode engine. |
| `indextts/s2mel/modules/flow_matching.py` | CFM (incl. `solve_single_step` for distilled students). |
| `scripts/serve_ha.py` | Orchestrator used by `start_ha.sh`. |
| `tools/wyoming_indextts.py` | Wyoming TCP server bridging HA → `/inference/stream`. |
| `tools/train_character_lora.py` | Current speaker trainer (stutter-weighted). |
| `tools/prepare_character_dataset.py` | Produces the JSONL + `stutter_mask` tensors the trainer consumes. |
| `tools/snapshot_cfm_teacher.py`, `build_reflow_manifest.py`, `generate_reflow_pairs.py`, `train_cfm_reflow.py`, `ab_eval_cfm.py` | The Phase-3 distillation pipeline. |
| `training/<speaker>/` | All per-speaker artifacts: dataset, character_lora/, reflow_manifest.jsonl, reflow_pairs/, cfm_reflow_student/, ab_results/. |
| `checkpoints/` | Base weights + `s2mel_distilled.pth` (active student) + per-speaker `s2mel_teacher_*.pth` snapshots. |
| `docs/` | Design docs. `STREAMING_LATENCY_RESULTS.md` and `HOMEASSISTANT_INTEGRATION_PLAN.md` are the load-bearing ones. |
| `archive/` | Old code. Don't reach in unless explicitly asked. |

---

## The production streaming path

Fast chunk mode with the distilled student is the default. The relevant knobs:

- **Distilled student active.** `checkpoints/s2mel_distilled.pth` is loaded
  automatically by `IndexTTS2.__init__` when present; only the CFM keys are
  overlaid on the base s2mel.
- **Preset.** `fast_quality` for short HA replies, `balanced_distilled` when
  you want sentence-paced output at indistinguishable-from-teacher quality.
  `ultra_fast` / `ultra_fast_distilled` exists but has a known OOD-shape
  quality issue documented in `STREAMING_LATENCY_RESULTS.md §5`.
- **Solver overrides.** `single_step` / 1 step / CFG 0.0 — these are baked
  into the `*_distilled` preset factories; no Inference-tab override needed.
- **Speaker-cond cache.** Keyed on the reference-audio path. Cold ≈ 14 ms,
  warm ≈ 2 ms. Prime it once after a fresh model load.
- **GPT/synth serialization gate (`gpt_can_proceed`).** Biggest single TTFA
  win (~400 ms). Pauses GPT decode while CFM is running so they don't fight
  for SMs. Don't remove it.
- **`use_decoded_hidden_states` is off by default.** It skips the redundant
  GPT-latent forward but the fused-attention path isn't proven bit-identical;
  A/B before flipping on.

If you're tempted to change any of these "for cleanliness," read §5 and §6 of
`STREAMING_LATENCY_RESULTS.md` first — most of these were arrived at by
measurement.

---

## Things to know before editing

- **Don't reintroduce the `max(12, diffusion_steps - 3)` floor** in
  `flow_matching.py` or anywhere else. Configs that request fewer steps win.
- **Audio output is 22050 Hz / 16-bit mono.** Wyoming, HA, and the WebUI
  player all assume this. Don't resample.
- **The Wyoming bridge exposes only speakers with a real character LoRA**
  (`gpt_lora_kind == "character"`). Don't add legacy verbatim/pattern speakers
  to HA — they were trained on a different input distribution and surprise
  users.
- **`prepare_character_dataset.py` is the *only* path that produces
  `stutter_mask`s.** If you add a new dataset prep tool, it must emit the
  mask or the trainer will silently train at uniform weight.
- **Per-chunk synth errors must be logged and skipped, not raised.** GPT
  errors are fatal; CFM/BigVGAN errors on chunk N shouldn't kill the rest of
  the stream. `_stream_by_tokens` in `streaming_v2.py` enforces this.
- **`release_cuda_cache_on_done=True`** in `streaming_inference_v2()` is what
  keeps long-running API processes from leaking the allocator pool. Keep it.
- **Don't add `@torch.compile` to `Sampler`** — it crashes CUDA-graph capture
  in `accel_engine.py`. This was removed deliberately.

---

## Common operations

```bash
# Train a new character voice (after putting audio under training/<name>/raw/)
python tools/prepare_character_dataset.py --speaker <name>
python tools/train_character_lora.py --speaker <name>
python tools/train_character_lora.py --speaker <name> --overfit-test   # sanity

# Distill the CFM for a speaker (long; use the WebUI Distillation tab in practice)
python tools/snapshot_cfm_teacher.py    --speaker <name> --merge-lora
python tools/build_reflow_manifest.py   --speaker <name>
python tools/generate_reflow_pairs.py   --speaker <name>
python tools/train_cfm_reflow.py        --speaker <name>
python tools/ab_eval_cfm.py             --speaker <name>
# then "Activate" the chosen best.pth via the WebUI to copy it into
# checkpoints/s2mel_distilled.pth and trigger a base-model reload.

# Diagnostics
curl -X POST http://localhost:8000/inference/warmup
curl      http://localhost:8000/system/gpu-stats
curl -X POST http://localhost:8000/system/gpu-cleanup
```
