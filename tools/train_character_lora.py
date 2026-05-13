#!/usr/bin/env python3
"""Character LoRA trainer for IndexTTS2.

Trains a GPT-side LoRA that teaches a speaker's *way of talking* — stutters,
fillers, pauses, false starts — while keeping the input text *clean* so the
upstream LLM in a voice-to-voice pipeline can write naturally.

Core mechanic that makes this work (vs the older verbatim trainer):
    The loss is weighted PER MEL-TOKEN, not per sample. A boolean
    `stutter_mask` from the prep step marks the mel-token positions that
    correspond to stutter / filler / repetition regions in the audio. The
    CE loss at those positions is scaled by `--stutter-weight` (default 15)
    so the gradient lands where the character actually lives, instead of
    being washed out by the surrounding clean tokens.

Built-in safety nets:
    --overfit-test  : trains for many epochs on just 2 samples to confirm
                      the loop can learn AT ALL. ~3 min sanity check.
    --logit-diff    : after training, dumps top-k mel-token probabilities
                      for a fixed prompt with vs without the LoRA so you
                      can SEE whether it changed anything before listening.

Saves to training/<speaker>/character_lora/{best,final}_checkpoint/lora/ — the
API's load-model path is taught to look here.

Usage:
    python tools/train_character_lora.py --speaker ozzyv5
    python tools/train_character_lora.py --speaker ozzyv5 --overfit-test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================== Dataset ==============================

@dataclass
class CharSample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    stutter_mask_path: Path
    text_len: int
    code_len: int
    audio_duration: float
    stutter_token_ratio: float
    clean_text: str
    verbatim_text: str


class CharacterDataset(Dataset):
    """Reads the JSONL manifest written by tools/prepare_character_dataset.py."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        self.samples: List[CharSample] = []
        self._load()

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.base_dir / path

    def _load(self):
        with open(self.manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                self.samples.append(CharSample(
                    id=r["id"],
                    text_ids_path=self._resolve(r["text_ids_path"]),
                    codes_path=self._resolve(r["codes_path"]),
                    condition_path=self._resolve(r["condition_path"]),
                    emo_vec_path=self._resolve(r["emo_vec_path"]),
                    stutter_mask_path=self._resolve(r["stutter_mask_path"]),
                    text_len=int(r["text_len"]),
                    code_len=int(r["code_len"]),
                    audio_duration=float(r.get("audio_duration", 0.0)),
                    stutter_token_ratio=float(r.get("stutter_token_ratio", 0.0)),
                    clean_text=r.get("clean_text", ""),
                    verbatim_text=r.get("verbatim_text", ""),
                ))
        n_stutter = sum(1 for s in self.samples if s.stutter_token_ratio > 0)
        avg_ratio = float(np.mean([s.stutter_token_ratio for s in self.samples])) if self.samples else 0.0
        print(f"[data] {len(self.samples)} samples ({n_stutter} with stutter regions, "
              f"avg stutter-token coverage {avg_ratio*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "id": s.id,
            "text_ids": torch.from_numpy(np.load(s.text_ids_path).astype(np.int64)),
            "codes": torch.from_numpy(np.load(s.codes_path).astype(np.int64)),
            "condition": torch.from_numpy(np.load(s.condition_path).astype(np.float32)),
            "emo_vec": torch.from_numpy(np.load(s.emo_vec_path).astype(np.float32)),
            "stutter_mask": torch.from_numpy(np.load(s.stutter_mask_path).astype(np.bool_)),
            "text_len": s.text_len,
            "code_len": s.code_len,
        }


def collate(batch: List[Dict]) -> Dict:
    text_padded = pad_sequence([b["text_ids"] for b in batch], batch_first=True, padding_value=0)
    code_padded = pad_sequence([b["codes"] for b in batch], batch_first=True, padding_value=0)
    mask_padded = pad_sequence([b["stutter_mask"] for b in batch], batch_first=True, padding_value=False)
    return {
        "ids": [b["id"] for b in batch],
        "text_ids": text_padded,
        "codes": code_padded,
        "stutter_mask": mask_padded,
        "condition": torch.stack([b["condition"] for b in batch], dim=0),
        "emo_vec": torch.stack([b["emo_vec"] for b in batch], dim=0),
        "text_lengths": torch.tensor([b["text_len"] for b in batch], dtype=torch.long),
        "code_lengths": torch.tensor([b["code_len"] for b in batch], dtype=torch.long),
    }


# ============================== Loss ==============================

def compute_loss(
    model: nn.Module,
    batch: Dict,
    device: torch.device,
    stutter_weight: float,
    text_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CE on (clean text → codes) with mel-token-level stutter mask weighting.

    stutter_weight applies to mel-token positions where stutter_mask is True.
    The clean tokens stay at weight 1.0. Final loss is the mean over weighted
    per-token CEs, normalized by the sum of weights so gradients don't blow
    up when batches have heavy stutter coverage.
    """
    base = model.base_model.model if hasattr(model, "base_model") else model

    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)
    stutter_mask = batch["stutter_mask"].to(device)  # (B, T_code) bool

    B = text_ids.size(0)

    # Build conditioning (mirrors inference path)
    use_speed = torch.zeros(B, dtype=torch.long, device=device)
    duration_ctrl = base.speed_emb(torch.ones_like(use_speed))
    duration_free = base.speed_emb(torch.zeros_like(use_speed))
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1),
         duration_ctrl.unsqueeze(1),
         duration_free.unsqueeze(1)),
        dim=1,
    )

    max_text = base.text_pos_embedding.emb.num_embeddings
    max_mel = base.mel_pos_embedding.emb.num_embeddings

    text_inputs = base.set_text_padding(text_ids.clone(), text_lengths)
    if text_inputs.size(1) + 2 > max_text:
        cap = max_text - 2
        text_inputs = text_inputs[:, :cap]
        text_lengths = torch.clamp(text_lengths, max=cap)
    text_inputs = F.pad(text_inputs, (0, 1), value=base.stop_text_token)
    text_inputs, text_targets = base.build_aligned_inputs_and_targets(
        text_inputs, base.start_text_token, base.stop_text_token
    )

    mel_inputs = base.set_mel_padding(codes.clone(), code_lengths)
    # Trim the stutter mask to match before we pad / shift to build targets.
    orig_T_code = stutter_mask.size(1)
    if mel_inputs.size(1) + 2 > max_mel:
        cap = max_mel - 2
        mel_inputs = mel_inputs[:, :cap]
        code_lengths = torch.clamp(code_lengths, max=cap)
        stutter_mask = stutter_mask[:, :cap]
    elif mel_inputs.size(1) > stutter_mask.size(1):
        # set_mel_padding may have shifted lengths; right-pad mask with False
        pad = mel_inputs.size(1) - stutter_mask.size(1)
        stutter_mask = F.pad(stutter_mask, (0, pad), value=False)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=base.stop_mel_token)
    # build_aligned_inputs_and_targets prepends start_mel_token to inputs and
    # appends stop_mel_token to targets — both targets and the corresponding
    # mask need to align with the (length+1) sequence.
    mel_inputs, mel_targets = base.build_aligned_inputs_and_targets(
        mel_inputs, base.start_mel_token, base.stop_mel_token
    )
    # Pad stutter mask on the RIGHT with False to match the target length
    if stutter_mask.size(1) < mel_targets.size(1):
        stutter_mask = F.pad(stutter_mask, (0, mel_targets.size(1) - stutter_mask.size(1)), value=False)
    else:
        stutter_mask = stutter_mask[:, :mel_targets.size(1)]

    text_emb = base.text_embedding(text_inputs) + base.text_pos_embedding(text_inputs)
    mel_emb = base.mel_embedding(mel_inputs) + base.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = base.get_logits(
        conds, text_emb, base.text_head, mel_emb, base.mel_head
    )

    # Length-derived presence mask
    mel_pos_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )
    text_pos_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )

    # Token-level weights — this is the structural change vs verbatim trainer
    token_weights = torch.where(
        stutter_mask, torch.tensor(stutter_weight, device=device),
        torch.tensor(1.0, device=device)
    )
    token_weights = token_weights * mel_pos_mask.float()

    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")
    mel_loss = (mel_ce * token_weights).sum() / token_weights.sum().clamp_min(1e-6)

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    text_loss = (text_ce * text_pos_mask.float()).sum() / text_pos_mask.sum().clamp_min(1)

    total = text_weight * text_loss + (1.0 - text_weight) * mel_loss

    # Metrics
    with torch.no_grad():
        mel_pred = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_tgt = mel_targets.reshape(-1)
        mel_pm = mel_pos_mask.reshape(-1)
        sm = stutter_mask.reshape(-1) & mel_pm
        if mel_pm.any():
            top1 = (mel_pred[mel_pm].argmax(-1) == mel_tgt[mel_pm]).float().mean().item()
        else:
            top1 = 0.0
        if sm.any():
            stutter_top1 = (mel_pred[sm].argmax(-1) == mel_tgt[sm]).float().mean().item()
        else:
            stutter_top1 = 0.0

    return total, {
        "loss": total.item(),
        "mel_loss": mel_loss.item(),
        "text_loss": text_loss.item(),
        "mel_top1": top1,
        "stutter_top1": stutter_top1,
        "stutter_tokens": int(sm.sum().item()),
        "total_tokens": int(mel_pm.sum().item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, stutter_weight):
    model.eval()
    losses = []
    stutter_top1s = []
    for batch in loader:
        _, metrics = compute_loss(model, batch, device, stutter_weight)
        losses.append(metrics["loss"])
        if metrics["stutter_tokens"] > 0:
            stutter_top1s.append(metrics["stutter_top1"])
    model.train()
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_stutter_top1": float(np.mean(stutter_top1s)) if stutter_top1s else float("nan"),
    }


# ============================== Diagnostic ==============================

# KL bands used by both the trainer's PASS/FAIL summary and the WebUI meter.
# Keep these in one place so the rendering stays consistent with the trainer.
KL_BANDS = [
    {"label": "no-op",  "min": 0.00, "max": 0.05, "color": "#d63031"},  # red
    {"label": "some",   "min": 0.05, "max": 0.50, "color": "#fdcb6e"},  # amber
    {"label": "real",   "min": 0.50, "max": 2.00, "color": "#00b894"},  # green
    {"label": "strong", "min": 2.00, "max": 8.00, "color": "#0984e3"},  # blue
]


def _kl_band(mean_kl: float) -> str:
    for band in KL_BANDS:
        if mean_kl < band["max"]:
            return band["label"]
    return KL_BANDS[-1]["label"]


def _build_sanity_check(
    mode: str,
    final_loss: Optional[float],
    final_stutter_top1: Optional[float],
    mean_kl: Optional[float],
) -> Dict:
    """Mode-aware PASS/FAIL with explicit reasons. Lives in the JSON so the
    WebUI can render a banner without re-deriving any thresholds.
    """
    reasons: List[str] = []
    passed = True

    if mode == "overfit":
        # Overfit-test on 2 samples should drive everything to the floor /
        # ceiling. If it doesn't, the loop or data pipeline is broken.
        if final_loss is None or final_loss > 0.5:
            passed = False
            reasons.append(f"loss did not converge below 0.5 (got {final_loss})")
        if final_stutter_top1 is None or final_stutter_top1 < 0.8:
            passed = False
            reasons.append(f"stutter_top1 below 0.8 on memorized data (got {final_stutter_top1}) — "
                           f"loss mask or stutter weighting probably not active")
        if mean_kl is None or mean_kl < 0.5:
            passed = False
            reasons.append(f"mean KL below 0.5 (got {mean_kl}) — LoRA didn't move the distribution")
    else:
        # Real training: we expect generalization, not memorization. The only
        # hard requirement is that the LoRA did something at all.
        if mean_kl is None or mean_kl < 0.05:
            passed = False
            reasons.append(f"mean KL below 0.05 (got {mean_kl}) — LoRA is essentially a no-op")
        elif mean_kl < 0.5:
            reasons.append(f"mean KL is moderate ({mean_kl:.3f}) — some learning but probably "
                           f"not enough to be clearly audible; consider more epochs or data")
        if final_loss is not None and final_loss > 5.0:
            passed = False
            reasons.append(f"loss is suspiciously high ({final_loss:.3f}) — training may not have converged")

    return {
        "mode": mode,
        "passed": passed,
        "reasons": reasons,
        "thresholds": {
            "overfit": {"loss_max": 0.5, "stutter_top1_min": 0.8, "kl_min": 0.5},
            "full":    {"kl_min": 0.05, "loss_max": 5.0},
        }[mode] if mode in ("overfit", "full") else None,
        "final_loss": final_loss,
        "final_stutter_top1": final_stutter_top1,
        "mean_kl": mean_kl,
        "kl_band": _kl_band(mean_kl) if mean_kl is not None else None,
    }


@torch.no_grad()
def logit_diff_diagnostic(
    model_with_lora: nn.Module,
    model_base: nn.Module,
    batch: Dict,
    device: torch.device,
    top_k: int = 8,
    out_path: Optional[Path] = None,
    sanity_check: Optional[Dict] = None,
) -> Dict:
    """Compare top-k mel-token distributions with vs without the LoRA on the
    same input. If they're nearly identical, the LoRA didn't learn anything
    no matter what the audio sounds like.
    """
    def _forward(m: nn.Module) -> torch.Tensor:
        base = m.base_model.model if hasattr(m, "base_model") else m
        condition = batch["condition"].to(device)
        text_ids = batch["text_ids"].to(device)
        codes = batch["codes"].to(device)
        emo_vec = batch["emo_vec"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        code_lengths = batch["code_lengths"].to(device)
        B = text_ids.size(0)
        use_speed = torch.zeros(B, dtype=torch.long, device=device)
        duration_ctrl = base.speed_emb(torch.ones_like(use_speed))
        duration_free = base.speed_emb(torch.zeros_like(use_speed))
        conds = torch.cat(
            (condition + emo_vec.unsqueeze(1),
             duration_ctrl.unsqueeze(1),
             duration_free.unsqueeze(1)),
            dim=1,
        )
        text_inputs = base.set_text_padding(text_ids.clone(), text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=base.stop_text_token)
        text_inputs, _ = base.build_aligned_inputs_and_targets(
            text_inputs, base.start_text_token, base.stop_text_token
        )
        mel_inputs = base.set_mel_padding(codes.clone(), code_lengths)
        mel_inputs = F.pad(mel_inputs, (0, 1), value=base.stop_mel_token)
        mel_inputs, _ = base.build_aligned_inputs_and_targets(
            mel_inputs, base.start_mel_token, base.stop_mel_token
        )
        text_emb = base.text_embedding(text_inputs) + base.text_pos_embedding(text_inputs)
        mel_emb = base.mel_embedding(mel_inputs) + base.mel_pos_embedding(mel_inputs)
        _, mel_logits = base.get_logits(
            conds, text_emb, base.text_head, mel_emb, base.mel_head
        )
        return mel_logits  # (B, V, T)

    logits_lora = F.softmax(_forward(model_with_lora), dim=1)
    logits_base = F.softmax(_forward(model_base), dim=1)

    # KL divergence per token + average — bigger = LoRA actually changed the
    # distribution. Near zero = LoRA is a no-op.
    eps = 1e-9
    kl = (logits_lora * (logits_lora.clamp_min(eps).log() - logits_base.clamp_min(eps).log())).sum(dim=1)
    mean_kl = float(kl.mean().item())
    median_kl = float(kl.median().item())
    max_kl = float(kl.max().item())

    # Sample one token's top-k for a human-readable peek
    samples = []
    for b in range(logits_lora.size(0)):
        # Use a mid-sequence position to dodge start-token noise
        t = min(logits_lora.size(2) - 1, max(0, logits_lora.size(2) // 3))
        base_topk = torch.topk(logits_base[b, :, t], top_k)
        lora_topk = torch.topk(logits_lora[b, :, t], top_k)
        samples.append({
            "sample_idx": b,
            "position": int(t),
            "base_top_ids": base_topk.indices.cpu().tolist(),
            "base_top_probs": [round(p, 4) for p in base_topk.values.cpu().tolist()],
            "lora_top_ids": lora_topk.indices.cpu().tolist(),
            "lora_top_probs": [round(p, 4) for p in lora_topk.values.cpu().tolist()],
        })

    report = {
        "mean_kl_divergence": mean_kl,
        "median_kl_divergence": median_kl,
        "max_kl_divergence": max_kl,
        "kl_band": _kl_band(mean_kl),
        "kl_bands": KL_BANDS,
        "interpretation": (
            "near zero (<0.05) = LoRA is a no-op" if mean_kl < 0.05
            else "moderate (0.05-0.5) = some learning" if mean_kl < 0.5
            else "strong (>0.5) = clear distribution shift"
        ),
        "samples": samples,
    }
    if sanity_check is not None:
        # Backfill mean_kl now that we have it
        sanity_check = dict(sanity_check)
        sanity_check["mean_kl"] = mean_kl
        sanity_check["kl_band"] = _kl_band(mean_kl)
        # Recompute pass/fail with KL available
        sanity_check = _build_sanity_check(
            sanity_check.get("mode", "full"),
            sanity_check.get("final_loss"),
            sanity_check.get("final_stutter_top1"),
            mean_kl,
        )
        report["sanity_check"] = sanity_check
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[diag] wrote {out_path}")
    return report


# ============================== Main ==============================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speaker", "-s", required=True)
    p.add_argument("--manifest", type=Path, default=None,
                   help="Defaults to training/<speaker>/character_dataset/manifest.jsonl")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to training/<speaker>/character_lora/")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "checkpoints" / "config.yaml")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--include-conditioning", action="store_true", default=True)
    p.add_argument("--include-heads", action="store_true", default=True)

    p.add_argument("--stutter-weight", type=float, default=15.0,
                   help="Token-level weight applied to stutter-mask positions in the CE loss.")
    p.add_argument("--text-weight", type=float, default=0.1,
                   help="Weight on the text-side CE term (mel is 1-text_weight).")

    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=2, help="Run validation every N epochs.")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=8, help="Early stop after N evals without improvement.")
    p.add_argument("--seed", type=int, default=42)

    # Modes
    p.add_argument("--overfit-test", action="store_true",
                   help="Sanity check: 2 samples, 200 epochs, no val split. Confirms the loop "
                        "can converge at all on this dataset/code path before burning a real run.")
    p.add_argument("--logit-diff", action="store_true",
                   help="After training, dump pre/post-LoRA top-k mel-token distributions to "
                        "<output-dir>/logit_diff.json.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gpt(cfg, gpt_path: Path, device: torch.device):
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    raw_state = torch.load(gpt_path, map_location="cpu").get("model", {})
    if "mel_pos_embedding.emb.weight" in raw_state:
        ckpt_dim = raw_state["mel_pos_embedding.emb.weight"].shape[1]
        if cfg.gpt.model_dim != ckpt_dim:
            cfg.gpt.model_dim = ckpt_dim
    del raw_state
    model = UnifiedVoice(**cfg.gpt)
    load_checkpoint(model, str(gpt_path))
    return model.to(device)


def main():
    args = parse_args()
    set_seed(args.seed)

    from indextts.utils.lora_utils import apply_lora_to_model, save_lora_checkpoint, get_trainable_parameters

    speaker_dir = PROJECT_ROOT / "training" / args.speaker
    manifest_path = args.manifest or (speaker_dir / "character_dataset" / "manifest.jsonl")
    if not manifest_path.exists():
        print(f"❌ manifest not found: {manifest_path}\n   run tools/prepare_character_dataset.py first", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_dir or (speaker_dir / "character_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={device}  manifest={manifest_path}")

    # === Build model ===
    cfg = OmegaConf.load(args.config)
    gpt_path = args.model_dir / cfg.gpt_checkpoint
    print(f"[train] loading base GPT from {gpt_path}")
    base_model_for_diag = None
    if args.logit_diff:
        # Keep a frozen-on-cpu copy for the diagnostic. We'll restore weights
        # into a fresh model after training rather than carrying two copies on
        # the GPU.
        base_model_for_diag = build_gpt(cfg, gpt_path, torch.device("cpu"))
        for p in base_model_for_diag.parameters():
            p.requires_grad_(False)
        base_model_for_diag.eval()

    model = build_gpt(cfg, gpt_path, device)
    model.train()

    print(f"[train] applying LoRA rank={args.lora_rank} alpha={args.lora_alpha}")
    model = apply_lora_to_model(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_gpt=True,
        include_conditioning=args.include_conditioning,
        include_heads=args.include_heads,
    )
    for name, param in model.named_parameters():
        if any(k in name for k in ["head", "lora", "emovec", "emo_layer"]):
            param.requires_grad = True

    # Enable input grads on embedding outputs (needed when grad checkpointing is on)
    def _make_outputs_require_grad(_m, _inp, output):
        output.requires_grad_(True)
    base = model.base_model.model if hasattr(model, "base_model") else model
    base.text_embedding.register_forward_hook(_make_outputs_require_grad)
    base.mel_embedding.register_forward_hook(_make_outputs_require_grad)

    stats = get_trainable_parameters(model)
    print(f"[train] trainable: {stats['trainable_params']:,} / {stats['all_params']:,} "
          f"({stats['trainable_percentage']:.2f}%)")

    # === Dataset ===
    dataset = CharacterDataset(manifest_path)
    if args.overfit_test:
        print("[train] OVERFIT TEST MODE — using first 2 samples, no val split")
        keep = min(2, len(dataset))
        dataset.samples = dataset.samples[:keep]
        args.epochs = max(args.epochs, 200)
        args.eval_every = 50
        args.patience = 999
        args.batch_size = min(args.batch_size, keep)
        val_dataset = None
    else:
        val_n = max(1, int(len(dataset) * args.val_frac)) if len(dataset) >= 10 else 0
        if val_n > 0:
            val_indices = list(range(len(dataset) - val_n, len(dataset)))
            train_indices = list(range(len(dataset) - val_n))
            val_dataset = CharacterDataset(manifest_path)
            val_dataset.samples = [dataset.samples[i] for i in val_indices]
            dataset.samples = [dataset.samples[i] for i in train_indices]
        else:
            val_dataset = None
        print(f"[train] split  train={len(dataset)}  val={len(val_dataset) if val_dataset else 0}")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    ) if val_dataset else None

    # === Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    total_steps = max(args.epochs * len(train_loader) // max(args.grad_accum, 1), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=min(0.3, args.warmup_steps / total_steps) if total_steps > 0 else 0.1,
        anneal_strategy="cos",
    )

    # === Train loop ===
    history = []
    best_val = float("inf")
    bad_evals = 0
    step = 0
    print(f"[train] {args.epochs} epochs, batch={args.batch_size}, "
          f"stutter_weight={args.stutter_weight}, total_steps={total_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_stutter_top1 = []
        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            loss, metrics = compute_loss(
                model, batch, device,
                stutter_weight=args.stutter_weight,
                text_weight=args.text_weight,
            )
            (loss / args.grad_accum).backward()
            epoch_losses.append(metrics["loss"])
            if metrics["stutter_tokens"] > 0:
                epoch_stutter_top1.append(metrics["stutter_top1"])

            if (batch_idx + 1) % args.grad_accum == 0 or batch_idx + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                if step < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_st = float(np.mean(epoch_stutter_top1)) if epoch_stutter_top1 else float("nan")
        dt = time.time() - t0
        log_line = (f"epoch {epoch:4d}/{args.epochs}  "
                    f"loss={epoch_loss:.4f}  stutter_top1={epoch_st:.3f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")
        print(log_line)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "stutter_top1": epoch_st})

        # Validation + early stop
        if val_loader and epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, args.stutter_weight)
            print(f"        val_loss={val_metrics['val_loss']:.4f}  "
                  f"val_stutter_top1={val_metrics['val_stutter_top1']:.3f}")
            history[-1].update(val_metrics)
            if val_metrics["val_loss"] < best_val - 1e-4:
                best_val = val_metrics["val_loss"]
                bad_evals = 0
                best_dir = output_dir / "best_checkpoint" / "lora"
                save_lora_checkpoint(model, best_dir, {
                    "epoch": epoch,
                    "val_loss": best_val,
                    "stutter_weight": args.stutter_weight,
                    "lora_rank": args.lora_rank,
                })
                print(f"        ✓ new best, saved to {best_dir}")
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"[train] early stop after {bad_evals} evals without improvement")
                    break

        if epoch % args.save_every == 0:
            ckpt = output_dir / "logs" / f"checkpoint_e{epoch:04d}" / "lora"
            save_lora_checkpoint(model, ckpt, {"epoch": epoch, "loss": epoch_loss})

    # === Final save ===
    final_dir = output_dir / "final_checkpoint" / "lora"
    save_lora_checkpoint(model, final_dir, {
        "speaker": args.speaker,
        "epochs_run": epoch,
        "final_loss": epoch_loss if epoch_losses else None,
        "stutter_weight": args.stutter_weight,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    })
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] final checkpoint saved to {final_dir}")

    # If we never produced a best_checkpoint (no val loader), copy final → best
    # so the API discovery path always finds something.
    best_dir = output_dir / "best_checkpoint" / "lora"
    if not (best_dir / "adapter_config.json").exists():
        import shutil
        best_dir.mkdir(parents=True, exist_ok=True)
        for f in final_dir.iterdir():
            shutil.copy2(f, best_dir / f.name)
        print(f"[train] mirrored final → best (no val split)")

    # === Logit diff diagnostic + sanity check ===
    mode = "overfit" if args.overfit_test else "full"
    final_loss_val = epoch_loss if epoch_losses else None
    final_stutter_top1_val = (
        float(np.mean(epoch_stutter_top1)) if epoch_stutter_top1 else None
    )
    sanity_seed = {
        "mode": mode,
        "final_loss": final_loss_val,
        "final_stutter_top1": final_stutter_top1_val,
        # mean_kl will be backfilled inside logit_diff_diagnostic when --logit-diff is on
        "mean_kl": None,
    }

    if args.logit_diff and len(dataset) > 0:
        print(f"[train] running logit-diff diagnostic...")
        # Reload base into a fresh GPT on the same device for forward parity
        base_for_diag = build_gpt(cfg, gpt_path, device)
        base_for_diag.eval()
        batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)),
                                     collate_fn=collate, shuffle=False)))
        report = logit_diff_diagnostic(
            model, base_for_diag, batch, device,
            top_k=8,
            out_path=output_dir / "logit_diff.json",
            sanity_check=sanity_seed,
        )
        print(f"[diag] mean KL = {report['mean_kl_divergence']:.4f}  → {report['interpretation']}")
        sc = report.get("sanity_check") or {}
        if sc.get("passed"):
            extra = f"  ({'; '.join(sc['reasons'])})" if sc.get("reasons") else ""
            print(f"[sanity] ✅ PASS [{mode}]  loss={final_loss_val}  "
                  f"stutter_top1={final_stutter_top1_val}  KL={report['mean_kl_divergence']:.4f}{extra}")
        else:
            print(f"[sanity] ❌ FAIL [{mode}]")
            for r in sc.get("reasons", []):
                print(f"          - {r}")
        del base_for_diag
    else:
        # No KL available — still emit a minimal sanity.json based on what we have.
        sc = _build_sanity_check(mode, final_loss_val, final_stutter_top1_val, None)
        (output_dir / "logit_diff.json").write_text(json.dumps({
            "mean_kl_divergence": None,
            "median_kl_divergence": None,
            "max_kl_divergence": None,
            "kl_band": None,
            "kl_bands": KL_BANDS,
            "interpretation": "skipped — re-run training with --logit-diff for distribution-shift metrics",
            "samples": [],
            "sanity_check": sc,
        }, indent=2))
        verdict = "✅ PASS" if sc["passed"] else "❌ FAIL"
        print(f"[sanity] {verdict} [{mode}] (KL not measured — pass with --logit-diff to compute)")

    print("[train] done.")


if __name__ == "__main__":
    main()
