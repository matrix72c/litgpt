#!/usr/bin/env python
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Evaluate a pretrained MoE model on English and Spanish validation sets separately.

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
import os
import sys
from pathlib import Path
from functools import partial
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from litgpt import Tokenizer
from litgpt.model import GPT, Config
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
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, max_iters: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for k, batch in enumerate(tqdm(dataloader, total=min(max_iters, len(dataloader)), desc="Evaluating")):
        if k >= max_iters:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        logits = model(input_ids)
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

    # ---- Evaluate ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("Evaluating on English validation set ...")
    en_loss = evaluate(model, en_dataloader, args.max_iters, device)
    en_ppl = math.exp(en_loss)
    print(f"  English  |  Loss: {en_loss:.4f}  |  PPL: {en_ppl:.2f}")

    print("\nEvaluating on Spanish validation set ...")
    es_loss = evaluate(model, es_dataloader, args.max_iters, device)
    es_ppl = math.exp(es_loss)
    print(f"  Spanish  |  Loss: {es_loss:.4f}  |  PPL: {es_ppl:.2f}")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"{'Language':<12} {'Loss':>10} {'PPL':>10}")
    print("-" * 34)
    print(f"{'English':<12} {en_loss:>10.4f} {en_ppl:>10.2f}")
    print(f"{'Spanish':<12} {es_loss:>10.4f} {es_ppl:>10.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
