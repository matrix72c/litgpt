# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class SafeData(DataModule):
    """Loads and mixes multiple StreamingDataset sources with configurable weights."""

    data_paths: Tuple[Union[str, Path], ...] = (Path("data/"),)
    """Paths to data directories containing preprocessed chunks for streaming datasets.
    Each path can also be a remote path (e.g., s3://). See also ``split_names`` if these paths contain subfolders
    for training- and validation splits."""
    weights: Optional[Tuple[float, ...]] = None
    """Optional weights for mixing datasets in ``data_paths``. If not provided, datasets are mixed uniformly."""
    harmful_data_paths: Optional[Tuple[Union[str, Path], ...]] = None
    """Subset of ``data_paths`` to label as harmful. When set, batches include an ``is_harmful`` field."""
    split_names: Optional[Tuple[str, str]] = None
    """Optional tuple for names of subfolders for training and validation under each ``data_paths`` entry.
    If not provided, all data under each path will be used for training, and the validation dataloader will be
    identical to the train dataloader."""
    batching_method: str = "stratified"
    """Batching method for CombinedStreamingDataset ("stratified" or "per_stream")."""
    iterate_over_all: bool = False
    """Whether to iterate over all datasets at least once per epoch in CombinedStreamingDataset."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)
    _harmful_data_path_set: set = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        super().__init__()
        if self.split_names is not None and len(self.split_names) != 2:
            raise ValueError("If provided `split_names` must be a tuple of two strings, for example: ('train', 'val').")
        if self.weights is not None and len(self.weights) != len(self.data_paths):
            raise ValueError("If provided `weights` must match the length of `data_paths`.")
        if self.harmful_data_paths is not None:
            data_path_set = {str(path) for path in self.data_paths}
            harmful_set = {str(path) for path in self.harmful_data_paths}
            unknown = sorted(harmful_set.difference(data_path_set))
            if unknown:
                raise ValueError("`harmful_data_paths` must be a subset of `data_paths`.")
            self._harmful_data_path_set = harmful_set

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(train=False)

    def _dataloader(self, train: bool) -> DataLoader:
        from litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader

        split_name = self.split_names[0] if train and self.split_names else None
        if not train and self.split_names:
            split_name = self.split_names[1]

        datasets = []
        for data_path in self.data_paths:
            input_dir = os.path.join(data_path, split_name) if split_name else str(data_path)
            transform = None
            if self.harmful_data_paths is not None:
                is_harmful = str(data_path) in self._harmful_data_path_set
                label = torch.tensor(1 if is_harmful else 0, dtype=torch.long)

                def _add_label(sample, label=label):
                    return {"input_ids": sample, "is_harmful": label}

                transform = _add_label
            dataset = StreamingDataset(
                input_dir=input_dir,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=train,
                seed=self.seed,
                drop_last=True,
                transform=transform,
            )
            datasets.append(dataset)

        combined_dataset = CombinedStreamingDataset(
            datasets=datasets,
            seed=self.seed,
            weights=self.weights,
            iterate_over_all=self.iterate_over_all,
            batching_method=self.batching_method,
        )

        return StreamingDataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )


class _HarmfulLabelDataset:
    def __init__(self, dataset, is_harmful: bool) -> None:
        self.dataset = dataset
        self.is_harmful = bool(is_harmful)

    def __iter__(self):
        label = torch.tensor(1 if self.is_harmful else 0, dtype=torch.long)
        for sample in self.dataset:
            yield {"input_ids": sample, "is_harmful": label}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)
