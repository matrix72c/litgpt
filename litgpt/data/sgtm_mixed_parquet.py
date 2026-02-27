# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""DataModule for SGTM that mixes multiple Parquet datasets with harmful labels."""

import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_parquet_texts(path: Union[str, Path], text_column: str) -> List[str]:
    """Read the text column from one or more Parquet files."""
    import pyarrow.parquet as pq

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found at: {path}")
    texts: List[str] = []
    for f in files:
        table = pq.read_table(f, columns=[text_column])
        texts.extend(table.column(text_column).to_pylist())
    return texts


def _tokenize(text: str, tokenizer: Tokenizer):
    """Tokenize a single text - used as the ``fn`` for ``litdata.optimize``."""
    yield tokenizer.encode(str(text), bos=False, eos=True)


def _dataset_name(path: Union[str, Path]) -> str:
    """Derive a human-readable dataset name from a file/directory path."""
    p = Path(path)
    name = p.stem if p.is_file() or p.suffix else p.name
    name = re.sub(r"[^\w\-]", "_", name).strip("_")
    return name or "dataset"


def _tokenizer_id(tokenizer: Tokenizer) -> str:
    """Return a short identifier for the tokenizer (model_name + vocab-size hash)."""
    model = getattr(tokenizer, "model_name", "unknown")
    vocab = str(getattr(tokenizer, "vocab_size", 0))
    digest = hashlib.md5(vocab.encode()).hexdigest()[:8]
    return f"{model}_{digest}"


def _available_cpu_count() -> int:
    """Return CPU count available to this process.

    Prefer CPU affinity when available to avoid `os.cpu_count()` under-reporting
    in some cgroup/container environments.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# SGTM Mixed Parquet DataModule
# ---------------------------------------------------------------------------


@dataclass
class SGTMMixedParquet(DataModule):
    """Loads and mixes multiple Parquet datasets for SGTM with harmful labels.

    This datamodule follows ``MixedParquet`` for preprocessing/caching and ``SafeData``
    for labeling behavior. Every returned sample includes:

    - ``input_ids``: token sequence
    - ``is_harmful``: 1 for harmful sources, 0 otherwise
    """

    data_paths: Tuple[Union[str, Path], ...] = (Path("data/"),)
    """Paths to Parquet files or directories of Parquet files used for training."""
    val_data_paths: Optional[Tuple[Optional[Union[str, Path]], ...]] = None
    """Optional per-source validation paths. Must match ``data_paths`` length when provided."""
    weights: Optional[Tuple[float, ...]] = None
    """Optional mixing weights, one per source. If not provided, sources are mixed uniformly."""
    harmful_data_paths: Optional[Tuple[Union[str, Path], ...]] = None
    """Subset of ``data_paths`` to label as harmful (``is_harmful=1``)."""
    text_column: str = "text"
    """Name of the Parquet column containing raw text."""
    cache_dir: Path = Path("data/.cache")
    """Root directory for tokenized litdata streaming caches."""
    val_split_fraction: float = 0.05
    """Fraction reserved for validation when no val path is provided for a source."""
    seed: int = 42
    """Random seed for shuffling."""
    num_workers: int = 8
    """Number of DataLoader / preprocessing workers."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    seq_length: int = field(default=2048, init=False, repr=False)
    _harmful_data_path_set: set = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        super().__init__()
        if not self.data_paths:
            raise ValueError("`data_paths` must contain at least one path.")
        if self.weights is not None and len(self.weights) != len(self.data_paths):
            raise ValueError("`weights` must match the length of `data_paths`.")
        if self.val_data_paths is not None and len(self.val_data_paths) != len(self.data_paths):
            raise ValueError("`val_data_paths` must match the length of `data_paths`.")
        if self.harmful_data_paths is not None:
            data_path_set = {str(path) for path in self.data_paths}
            harmful_set = {str(path) for path in self.harmful_data_paths}
            unknown = sorted(harmful_set.difference(data_path_set))
            if unknown:
                raise ValueError("`harmful_data_paths` must be a subset of `data_paths`.")
            self._harmful_data_path_set = harmful_set

    def _source_cache_dir(self, data_path: Union[str, Path]) -> Path:
        """Return the cache directory for a source, e.g. ``.cache/wiki_llama3_a1b2c3d4/``."""
        name = _dataset_name(data_path)
        tok_id = _tokenizer_id(self.tokenizer)
        return self.cache_dir / f"{name}_{tok_id}"

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = (max_seq_length or 2048) + 1

    def prepare_data(self) -> None:
        """Tokenize each Parquet source and write litdata streaming chunks to cache."""
        from litdata import optimize
        from litdata.streaming import TokensLoader

        for i, data_path in enumerate(self.data_paths):
            cache = self._source_cache_dir(data_path)
            train_out = cache / "train"
            val_out = cache / "val"

            if train_out.is_dir() and val_out.is_dir():
                print(f"[SGTMMixedParquet] Cache hit for '{_dataset_name(data_path)}' at {cache}, skipping.")
                continue

            print(f"[SGTMMixedParquet] Preparing source '{_dataset_name(data_path)}' -> {cache}")

            train_texts = _read_parquet_texts(data_path, self.text_column)

            val_path = self.val_data_paths[i] if self.val_data_paths is not None else None
            if val_path is not None:
                val_texts = _read_parquet_texts(val_path, self.text_column)
            else:
                n_val = max(1, int(len(train_texts) * self.val_split_fraction))
                val_texts = train_texts[-n_val:]
                train_texts = train_texts[:-n_val]

            num_proc = max(1, _available_cpu_count() - 1)

            if not train_out.is_dir():
                optimize(
                    fn=partial(_tokenize, tokenizer=self.tokenizer),
                    inputs=train_texts,
                    output_dir=str(train_out),
                    num_workers=min(num_proc, len(train_texts)),
                    chunk_bytes="200MB",
                    item_loader=TokensLoader(block_size=self.seq_length),
                )

            if not val_out.is_dir():
                optimize(
                    fn=partial(_tokenize, tokenizer=self.tokenizer),
                    inputs=val_texts,
                    output_dir=str(val_out),
                    num_workers=min(num_proc, len(val_texts)),
                    chunk_bytes="200MB",
                    item_loader=TokensLoader(block_size=self.seq_length),
                )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(train=False)

    def _dataloader(self, train: bool) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        split_name = "train" if train else "val"
        datasets = []

        for data_path in self.data_paths:
            source_dir = self._source_cache_dir(data_path) / split_name
            if not source_dir.is_dir():
                if train:
                    raise FileNotFoundError(
                        f"Training cache not found for source '{data_path}' at '{source_dir}'. Run prepare_data first."
                    )
                continue

            transform = None
            if self.harmful_data_paths is not None:
                is_harmful = str(data_path) in self._harmful_data_path_set
                label = torch.tensor(1 if is_harmful else 0, dtype=torch.long)

                def _add_label(sample, label=label):
                    return {"input_ids": sample, "is_harmful": label}

                transform = _add_label

            ds = StreamingDataset(
                input_dir=str(source_dir),
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=train,
                seed=self.seed,
                drop_last=True,
                transform=transform,
            )
            datasets.append(ds)

        if not datasets:
            raise RuntimeError(f"No {split_name} datasets found. Run prepare_data and verify data paths.")

        if len(datasets) == 1:
            data = datasets[0]
        else:
            w = self.weights if self.weights is not None else tuple(1.0 for _ in datasets)
            data = CombinedStreamingDataset(datasets=datasets, seed=self.seed, weights=w, iterate_over_all=False)

        return StreamingDataLoader(
            data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
