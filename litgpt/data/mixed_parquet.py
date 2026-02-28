# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""DataModule that mixes multiple Parquet datasets (pretraining) according to configurable ratios."""

import hashlib
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    """Tokenize a single text – used as the ``fn`` for ``litdata.optimize``."""
    yield tokenizer.encode(str(text), bos=False, eos=True)


def _cache_key(data_path: Union[str, Path], tokenizer_path: str) -> str:
    """Return an 8-char hex hash of the dataset path + tokenizer path."""
    raw = f"{data_path}|{tokenizer_path}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


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
# MixedParquet DataModule
# ---------------------------------------------------------------------------

@dataclass
class MixedParquet(DataModule):
    """Loads and mixes multiple Parquet datasets for **pretraining**.

    Each source is identified by a path in ``data_paths``.  Paths point to ``.parquet``
    files or directories of ``.parquet`` files.  Raw text is tokenized during
    ``prepare_data()`` and cached under ``.cache/<dataset_name>_<tokenizer_id>/``.
    The dataloaders use ``CombinedStreamingDataset`` to mix sources according to
    ``weights``, following the same pattern as :class:`TinyLlama` and :class:`SafeData`.

    **Caching**: each source is cached independently at::

        .cache/<dataset_name>_<tokenizer_id>/train/
        .cache/<dataset_name>_<tokenizer_id>/val/

    where ``<dataset_name>`` is derived from the file/directory name and
    ``<tokenizer_id>`` is ``<model_name>_<vocab_hash>``.  Switching tokenizers
    automatically invalidates stale caches.

    **Validation split logic (per source)**:

    * If ``val_data_paths`` is provided and the corresponding entry is not ``null`` →
      validation data comes from that path; no automatic split for that source.
    * Otherwise → the last ``val_split_fraction`` of documents is carved off for
      validation.

    Example YAML config::

        data:
          class_path: litgpt.data.MixedParquet
          init_args:
            data_paths:
              - data/wiki.parquet
              - data/code/
            val_data_paths:
              - null
              - data/code_val/
            weights:
              - 0.7
              - 0.3
            text_column: text
    """

    data_paths: Tuple[Union[str, Path], ...] = (Path("data/"),)
    """Paths to Parquet files or directories of Parquet files used for training.
    Each path is a separate data source."""
    val_data_paths: Optional[Tuple[Optional[Union[str, Path]], ...]] = None
    """Optional per-source validation paths.  Must match the length of ``data_paths``.
    Set an entry to ``null`` / ``None`` to auto-split that source instead."""
    weights: Optional[Tuple[float, ...]] = None
    """Optional mixing weights, one per source.  If not provided, sources are mixed uniformly."""
    text_column: str = "text"
    """Name of the Parquet column containing raw text.  Applied to all sources."""
    cache_dir: Path = Path(".cache")
    """Root directory for tokenized litdata streaming caches."""
    val_split_fraction: float = 0.05
    """Fraction of documents reserved for validation when no val path is given for a source."""
    seed: int = 42
    """Random seed for shuffling."""
    num_workers: int = 8
    """Number of DataLoader / preprocessing workers."""

    # -- runtime fields (set via connect()) --
    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    seq_length: int = field(default=2048, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if not self.data_paths:
            raise ValueError("`data_paths` must contain at least one path.")
        if self.weights is not None and len(self.weights) != len(self.data_paths):
            raise ValueError("`weights` must match the length of `data_paths`.")
        if self.val_data_paths is not None and len(self.val_data_paths) != len(self.data_paths):
            raise ValueError("`val_data_paths` must match the length of `data_paths`.")

    # ------------------------------------------------------------------
    # Cache path helpers
    # ------------------------------------------------------------------

    def _source_cache_dir(self, data_path: Union[str, Path]) -> Path:
        """Return the cache directory for a source, e.g. ``.cache/a1b2c3d4/``."""
        tok_path = str(getattr(self.tokenizer, "path", ""))
        key = _cache_key(data_path, tok_path)
        return self.cache_dir / key

    # ------------------------------------------------------------------
    # DataModule interface
    # ------------------------------------------------------------------

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = (max_seq_length or 2048) + 1  # +1 for next-token target

    def prepare_data(self) -> None:
        """Tokenize each Parquet source and write litdata streaming chunks to cache."""
        from litdata import optimize

        for i, data_path in enumerate(self.data_paths):
            cache = self._source_cache_dir(data_path)
            train_out = cache / "train"
            val_out = cache / "val"

            if train_out.is_dir() and val_out.is_dir():
                print(f"[MixedParquet] Cache hit for '{data_path}' at {cache}, skipping.")
                continue

            print(f"[MixedParquet] Preparing source '{data_path}' → {cache}")

            # --- read raw texts ---
            train_texts = _read_parquet_texts(data_path, self.text_column)

            val_path = self.val_data_paths[i] if self.val_data_paths is not None else None
            if val_path is not None:
                val_texts = _read_parquet_texts(val_path, self.text_column)
            else:
                n_val = max(1, int(len(train_texts) * self.val_split_fraction))
                val_texts = train_texts[-n_val:]
                train_texts = train_texts[:-n_val]

            # --- tokenize & write ---
            num_proc = max(1, _available_cpu_count() - 1)

            if not train_out.is_dir():
                optimize(
                    fn=partial(_tokenize, tokenizer=self.tokenizer),
                    inputs=train_texts,
                    output_dir=str(train_out),
                    num_workers=min(num_proc, len(train_texts)),
                    chunk_bytes="200MB",
                )

            if not val_out.is_dir():
                optimize(
                    fn=partial(_tokenize, tokenizer=self.tokenizer),
                    inputs=val_texts,
                    output_dir=str(val_out),
                    num_workers=min(num_proc, len(val_texts)),
                    chunk_bytes="200MB",
                )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        datasets = []
        for data_path in self.data_paths:
            cache = self._source_cache_dir(data_path)
            ds = StreamingDataset(
                input_dir=str(cache / "train"),
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
            datasets.append(ds)

        if len(datasets) == 1:
            train_data = datasets[0]
        else:
            w = self.weights if self.weights is not None else tuple(1.0 for _ in datasets)
            train_data = CombinedStreamingDataset(
                datasets=datasets, seed=self.seed, weights=w, iterate_over_all=False
            )

        return StreamingDataLoader(
            train_data, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        datasets = []
        for data_path in self.data_paths:
            val_dir = self._source_cache_dir(data_path) / "val"
            if val_dir.is_dir():
                ds = StreamingDataset(
                    input_dir=str(val_dir),
                    item_loader=TokensLoader(block_size=self.seq_length),
                    shuffle=False,
                )
                datasets.append(ds)

        if not datasets:
            return self.train_dataloader()

        if len(datasets) == 1:
            val_data = datasets[0]
        else:
            val_data = CombinedStreamingDataset(datasets=datasets, seed=self.seed, iterate_over_all=False)

        return StreamingDataLoader(
            val_data, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
