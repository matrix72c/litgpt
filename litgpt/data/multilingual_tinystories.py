# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Multilingual TinyStories DataModule.

Reads a parquet dataset with 'english' and 'spanish' columns, shuffles the two
languages independently, interleaves them, and creates a streaming token dataset
for pretraining.  Because the two language versions of every original row are
placed at independently random positions in the token stream, the probability
that they end up in the same mini-batch is negligible (~batch_size / 2N).
"""

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from litgpt.data.base import DataModule
from litgpt.data.text_files import validate_tokenizer
from litgpt.tokenizer import Tokenizer


@dataclass
class MultilingualTinyStories(DataModule):
    """Multilingual TinyStories data module.

    Expects a directory containing ``train.parquet`` and ``validation.parquet``,
    each with ``english`` and ``spanish`` text columns.  The two languages are
    shuffled independently and interleaved so that paired translations are
    extremely unlikely to appear in the same batch.

    Provides training and validation dataloaders that return batches of packed
    token blocks (suitable for language-model pretraining).
    """

    data_path: Path = Path("data/multilingual-tinystories")
    """Path to the directory that contains train.parquet and validation.parquet."""
    seed: int = 42
    """Random seed used for shuffling."""
    num_workers: int = 8
    """Number of dataloader workers."""
    num_shards: int = 20
    """Number of JSON shard files to create during preprocessing (controls parallelism)."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: int = -1,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # +1 because we need the next token as target

        # Build cache paths that include a tokenizer fingerprint so that
        # switching tokenizers does NOT silently reuse stale cached tokens.
        tok_id = _tokenizer_id(tokenizer)
        self.data_path_train = self.data_path / f"train_processed_{tok_id}"
        self.data_path_val = self.data_path / f"val_processed_{tok_id}"

    # ------------------------------------------------------------------
    # Data preparation (runs once, on rank 0)
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        from litdata import TokensLoader, optimize

        num_cpu = max(os.cpu_count() - 1, 1)

        # --- Training split ------------------------------------------------
        if not Path(self.data_path_train).is_dir():
            train_shard_dir = self.data_path / "train_shards"
            if not train_shard_dir.is_dir():
                import pandas as pd

                print("Reading train.parquet …")
                df = pd.read_parquet(self.data_path / "train.parquet")
                _write_shuffled_shards(df, train_shard_dir, self.seed, self.num_shards)
                del df

            shard_files = sorted(glob.glob(str(train_shard_dir / "*.json")))
            assert len(shard_files) > 0, f"No shard files found in {train_shard_dir}"
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=shard_files,
                output_dir=str(self.data_path_train),
                num_workers=min(num_cpu, len(shard_files)),
                chunk_bytes="200MB",
                item_loader=TokensLoader(),
            )

        # --- Validation split ----------------------------------------------
        if not Path(self.data_path_val).is_dir():
            val_shard_dir = self.data_path / "val_shards"
            if not val_shard_dir.is_dir():
                import pandas as pd

                print("Reading validation.parquet …")
                df = pd.read_parquet(self.data_path / "validation.parquet")
                _write_shuffled_shards(df, val_shard_dir, self.seed + 1, max(self.num_shards // 10, 1))
                del df

            shard_files = sorted(glob.glob(str(val_shard_dir / "*.json")))
            assert len(shard_files) > 0, f"No shard files found in {val_shard_dir}"
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=shard_files,
                output_dir=str(self.data_path_val),
                num_workers=min(num_cpu, len(shard_files)),
                chunk_bytes="200MB",
                item_loader=TokensLoader(),
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.data_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        return StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.data_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        return StreamingDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )


# ======================================================================
# Helper functions
# ======================================================================


def _write_shuffled_shards(
    df,
    shard_dir: Path,
    seed: int,
    num_shards: int,
) -> None:
    """Shuffle English and Spanish texts independently, interleave, and write to
    JSON shard files.

    Each shard is a JSON list of strings (plain text).  Independent shuffling
    ensures that paired translations are placed at uncorrelated positions in the
    combined sequence, making same-batch collocation extremely unlikely.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    en_texts = df["english"].tolist()
    es_texts = df["spanish"].tolist()
    n = len(en_texts)

    rng = np.random.RandomState(seed)
    en_order = rng.permutation(n)
    # Use a different sub-seed so the two permutations are independent
    es_order = np.random.RandomState(seed + 12345).permutation(n)

    # Interleave: en[0], es[0], en[1], es[1], …
    all_texts: list[str] = []
    for i in range(n):
        all_texts.append(en_texts[en_order[i]])
        all_texts.append(es_texts[es_order[i]])

    total = len(all_texts)
    shard_size = total // num_shards

    print(f"Writing {num_shards} shards ({total} texts) to {shard_dir} …")
    for s in range(num_shards):
        start = s * shard_size
        end = start + shard_size if s < num_shards - 1 else total
        path = shard_dir / f"shard_{s:05d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_texts[start:end], f, ensure_ascii=False)
    print("Done writing shards.")


def tokenize(filename: str, tokenizer: Tokenizer):
    """Read a JSON shard (list of plain-text strings) and yield token arrays."""
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    global_rank = int(os.environ.get("DATA_OPTIMIZER_GLOBAL_RANK", 0))
    num_workers = int(os.environ.get("DATA_OPTIMIZER_NUM_WORKERS", 1))
    local_rank = global_rank % num_workers

    for text in tqdm(data, position=local_rank):
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


def _tokenizer_id(tokenizer: Optional[Tokenizer]) -> str:
    """Return a short deterministic hash that identifies a tokenizer.

    The hash is derived from the tokenizer's vocab size and a sample encoding so
    that different tokenizers produce different cache directory names.
    """
    if tokenizer is None:
        return "none"
    # Use vocab size + encoding of a fixed probe string as fingerprint
    probe = "Hello 你好 Hola"
    try:
        encoded = tokenizer.encode(probe, bos=False, eos=False).tolist()
    except Exception:
        encoded = []
    raw = f"{tokenizer.vocab_size}_{encoded}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
