#!/usr/bin/env python
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Evaluate a pretrained MoE model on English and Spanish validation sets separately,
and observe the token load distribution across safe vs standard experts.

Usage:
    python scripts/tinystories/eval_multilingual.py \
        --checkpoint_dir checkpoints/Qwen-TinyStories-MoE-Base/final \
        --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base \
        --data_path data/multilingual-tinystories \
        --batch_size 32 \
        --max_seq_length 1024 \
        --max_iters 100
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litgpt import Tokenizer
from litgpt.model import GPT, Config, LLaMAMoE
from litgpt.utils import chunked_cross_entropy


# ---------------------------------------------------------------------------
# Simple in-memory dataset: tokenize texts and pack into fixed-length blocks
# ---------------------------------------------------------------------------

class TextBlockDataset(Dataset):
    """Tokenize a list of texts, concatenate all tokens, and chop into
    fixed-length blocks of `block_size + 1` (the extra token is the target for
    the last position)."""

    def __init__(self, texts: List[str], tokenizer: Tokenizer, block_size: int):
        all_tokens: List[int] = []
        for text in tqdm(texts, desc="Tokenizing", leave=False):
            text = text.strip()
            if not text:
                continue
            tokens = tokenizer.encode(text, bos=True, eos=False).tolist()
            all_tokens.extend(tokens)

        seq_len = block_size + 1  # input + 1 target token
        n_blocks = len(all_tokens) // seq_len
        # Truncate to exact multiple
        all_tokens = all_tokens[: n_blocks * seq_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).reshape(n_blocks, seq_len)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Expert load tracker
# ---------------------------------------------------------------------------

def _compute_cv(counts: List[int]) -> float:
    """Compute coefficient of variation (CV = std/mean) for a list of counts.
    CV=0 means perfectly balanced, higher values indicate more imbalance."""
    total = sum(counts)
    if total == 0:
        return 0.0
    mean = total / len(counts)
    if mean == 0:
        return 0.0
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    return (variance ** 0.5) / mean


def _compute_balance_metrics(counts: List[int]) -> Dict[str, float]:
    """Compute multiple balance metrics for expert load distribution."""
    total = sum(counts)
    if total == 0 or len(counts) == 0:
        return {"cv": 0.0, "max_min_ratio": 0.0, "min_pct": 0.0}

    mean = total / len(counts)
    cv = _compute_cv(counts)
    max_count = max(counts)
    min_count = min(counts)

    # Max/min ratio (1.0 means perfectly balanced)
    max_min_ratio = max_count / min_count if min_count > 0 else float("inf")

    # Minimum usage percentage
    min_pct = 100.0 * min_count / total

    return {"cv": cv, "max_min_ratio": max_min_ratio, "min_pct": min_pct}


class ExpertLoadTracker:
    """Register forward hooks on all MoE layers to count how many tokens are
    routed to each expert (standard vs safe), per layer."""

    def __init__(self, model: GPT):
        self.config = model.config
        # stats[layer_idx]["standard"][expert_idx] = token_count
        # stats[layer_idx]["safe"][expert_idx] = token_count
        self.stats: Dict[int, Dict[str, List[int]]] = {}
        self._hooks = []
        self._register(model)

    def _register(self, model: GPT):
        for layer_idx, block in enumerate(model.transformer.h):
            mlp = block.mlp
            if not isinstance(mlp, LLaMAMoE):
                continue
            n_expert = self.config.n_expert
            n_safe = self.config.n_safe_expert or 0
            self.stats[layer_idx] = {
                "standard": [0] * n_expert,
                "safe": [0] * n_safe,
            }
            for expert_idx, expert in enumerate(mlp.experts):
                hook = expert.register_forward_hook(
                    self._make_hook(layer_idx, "standard", expert_idx)
                )
                self._hooks.append(hook)
            if n_safe > 0:
                for expert_idx, expert in enumerate(mlp.safe_experts):
                    hook = expert.register_forward_hook(
                        self._make_hook(layer_idx, "safe", expert_idx)
                    )
                    self._hooks.append(hook)

    def _make_hook(self, layer_idx: int, expert_type: str, expert_idx: int):
        def hook(module, args, output):
            # args[0] shape: (num_routed_tokens, hidden_size)
            num_tokens = args[0].shape[0]
            self.stats[layer_idx][expert_type][expert_idx] += num_tokens
        return hook

    def reset(self):
        for layer_stats in self.stats.values():
            for expert_type in layer_stats:
                for i in range(len(layer_stats[expert_type])):
                    layer_stats[expert_type][i] = 0

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def print_summary(self, lang: str):
        """Print per-layer and aggregated expert load statistics."""
        n_expert = self.config.n_expert
        n_safe = self.config.n_safe_expert or 0
        total_standard = [0] * n_expert
        total_safe = [0] * n_safe

        print(f"\n{'='*70}")
        print(f"  Expert Load Distribution — {lang.upper()}")
        print(f"{'='*70}")

        for layer_idx in sorted(self.stats.keys()):
            layer = self.stats[layer_idx]
            std_counts = layer["standard"]
            safe_counts = layer["safe"]
            std_total = sum(std_counts)
            safe_total = sum(safe_counts)
            layer_total = std_total + safe_total

            for i, c in enumerate(std_counts):
                total_standard[i] += c
            for i, c in enumerate(safe_counts):
                total_safe[i] += c

            std_pct = 100.0 * std_total / layer_total if layer_total > 0 else 0
            safe_pct = 100.0 * safe_total / layer_total if layer_total > 0 else 0

            # Internal balance metrics per layer
            std_cv_str = ""
            safe_cv_str = ""
            if n_expert > 1 and std_total > 0:
                std_cv = _compute_cv(std_counts)
                std_cv_str = f" (CV={std_cv:.2f})"
            if n_safe > 1 and safe_total > 0:
                safe_cv = _compute_cv(safe_counts)
                safe_cv_str = f" (CV={safe_cv:.2f})"

            print(f"  Layer {layer_idx:>2}  |  standard: {std_total:>8} ({std_pct:5.1f}%){std_cv_str}  "
                  f"|  safe: {safe_total:>8} ({safe_pct:5.1f}%){safe_cv_str}")

        # Aggregated summary
        agg_std = sum(total_standard)
        agg_safe = sum(total_safe)
        agg_total = agg_std + agg_safe
        std_pct = 100.0 * agg_std / agg_total if agg_total > 0 else 0
        safe_pct = 100.0 * agg_safe / agg_total if agg_total > 0 else 0

        # Balance metrics
        std_cv = 0.0
        safe_cv = 0.0
        if n_expert > 0:
            std_metrics = _compute_balance_metrics(total_standard)
            std_cv = std_metrics['cv']
        if n_safe > 0:
            safe_metrics = _compute_balance_metrics(total_safe)
            safe_cv = safe_metrics['cv']

        # Print summary
        print(f"\n  TOTAL: standard={agg_std} ({std_pct:.1f}%), safe={agg_safe} ({safe_pct:.1f}%)")
        print(f"  Balance CV: std={std_cv:.3f}, safe={safe_cv:.3f}")
        print(f"{'=' * 70}")

        # Return metrics for summary line
        return {
            "std_total": agg_std,
            "safe_total": agg_safe,
            "std_pct": std_pct,
            "safe_pct": safe_pct,
            "std_cv": std_cv,
            "safe_cv": safe_cv,
        }


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    device: torch.device,
    exclude_safe_experts: bool = False,
) -> float:
    model.eval()
    losses = []
    for k, batch in enumerate(tqdm(dataloader, total=min(max_iters, len(dataloader)), desc="Evaluating")):
        if k >= max_iters:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        # Default: use all experts (including safe experts)
        # When exclude_safe_experts=True: only use standard experts
        logits = model(input_ids, exclude_safe_experts=exclude_safe_experts)

        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on English and Spanish validation sets")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/Qwen-TinyStories-MoE-Base/final",
                        help="Path to checkpoint directory (contains lit_model.pth and model_config.yaml)")
    parser.add_argument("--tokenizer_dir", type=str, default="checkpoints/Qwen3-30B-A3B-Base",
                        help="Path to tokenizer directory")
    parser.add_argument("--data_path", type=str, default="data/multilingual-tinystories",
                        help="Path to multilingual-tinystories data directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Maximum number of batches to evaluate per language")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"],
                        help="Inference precision")
    parser.add_argument("--exclude_safe_experts", action="store_true",
                        help="Exclude safe experts from routing (use only standard experts)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_path = Path(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    # ---- Load model --------------------------------------------------------
    print(f"Loading model from {checkpoint_dir} ...")
    config = Config.from_checkpoint(checkpoint_dir)
    model = GPT(config)

    checkpoint_file = checkpoint_dir / "lit_model.pth"
    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    # fabric.save wraps model state under "model" key
    if "model" in state:
        model_state = state["model"]
    else:
        model_state = state
    model.load_state_dict(model_state, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_expert={config.n_expert}, n_safe_expert={config.n_safe_expert}, "
          f"n_expert_per_token={config.n_expert_per_token}")

    # ---- Register expert load hooks ----------------------------------------
    tracker = ExpertLoadTracker(model)
    print(f"Expert load hooks registered on {len(tracker.stats)} MoE layers")

    # ---- Load tokenizer ----------------------------------------------------
    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer = Tokenizer(tokenizer_dir)
    print(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # ---- Load validation texts per language --------------------------------
    en_val_path = data_path / "en" / "validation.parquet"
    es_val_path = data_path / "es" / "validation.parquet"

    if not en_val_path.exists():
        raise FileNotFoundError(f"English validation file not found at {en_val_path}")
    if not es_val_path.exists():
        raise FileNotFoundError(f"Spanish validation file not found at {es_val_path}")

    print("Loading validation texts ...")
    english_texts = pd.read_parquet(en_val_path, columns=["text"])["text"].astype(str).tolist()
    spanish_texts = pd.read_parquet(es_val_path, columns=["text"])["text"].astype(str).tolist()

    print(f"  English texts: {len(english_texts)}")
    print(f"  Spanish texts: {len(spanish_texts)}")

    # ---- Build per-language datasets and dataloaders -----------------------
    block_size = args.max_seq_length

    print("\nPreparing English validation dataset ...")
    en_dataset = TextBlockDataset(english_texts, tokenizer, block_size)
    en_dataloader = DataLoader(en_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"  English blocks: {len(en_dataset)}, batches: {len(en_dataloader)}")

    print("Preparing Spanish validation dataset ...")
    es_dataset = TextBlockDataset(spanish_texts, tokenizer, block_size)
    es_dataloader = DataLoader(es_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"  Spanish blocks: {len(es_dataset)}, batches: {len(es_dataloader)}")

    # ---- Evaluate English --------------------------------------------------
    print("\n" + "=" * 50)
    print("Evaluating on English validation set ...")
    tracker.reset()
    en_loss = evaluate(model, en_dataloader, args.max_iters, device,
                       exclude_safe_experts=args.exclude_safe_experts)
    en_ppl = math.exp(en_loss)
    print(f"  English  |  Loss: {en_loss:.4f}  |  PPL: {en_ppl:.2f}")
    en_metrics = tracker.print_summary("English")

    # ---- Evaluate Spanish --------------------------------------------------
    print("\nEvaluating on Spanish validation set ...")
    tracker.reset()
    es_loss = evaluate(model, es_dataloader, args.max_iters, device,
                       exclude_safe_experts=args.exclude_safe_experts)
    es_ppl = math.exp(es_loss)
    print(f"  Spanish  |  Loss: {es_loss:.4f}  |  PPL: {es_ppl:.2f}")
    es_metrics = tracker.print_summary("Spanish")

    # ---- Cleanup hooks -----------------------------------------------------
    tracker.remove_hooks()

    # ---- One-line Summary (copy-paste friendly) ----------------------------
    exclude_str = "_exclude_safe" if args.exclude_safe_experts else ""
    print(f"\n| Config | en_loss | en_ppl | es_loss | es_ppl | std_pct | safe_pct | std_cv | safe_cv |")
    print(f"|--------|---------|--------|---------|--------|---------|---------|--------|---------|")
    print(f"|{exclude_str}|{en_loss:.4f}|{en_ppl:.2f}|{es_loss:.4f}|{es_ppl:.2f}|{en_metrics['std_pct']:.1f}%|{en_metrics['safe_pct']:.1f}%|{en_metrics['std_cv']:.3f}|{en_metrics['safe_cv']:.3f}|")


if __name__ == "__main__":
    main()
