"""Expand the number of **safe** experts in a MoE litgpt checkpoint.

New safe experts are initialized by randomly copying weights from the existing
*regular* experts in the same layer.  A ``safe_gate`` weight matrix is created
(or expanded) accordingly — each new row is copied from the regular ``gate``
row of the same source expert.

Usage:
    litgpt expand_moe_experts \
        --checkpoint_dir checkpoints/Qwen/Qwen3-30B-A3B \
        --output_dir checkpoints/Qwen/Qwen3-30B-A3B-safe \
        --num_new_safe_experts 64
"""

import re
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from litgpt.config import Config
from litgpt.utils import extend_checkpoint_dir, incremental_save, lazy_load, save_config


@torch.inference_mode()
def expand_moe_experts(
    *,
    checkpoint_dir: Path,
    num_new_safe_experts: int,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    dtype: Optional[str] = None,
) -> None:
    """Expand the number of safe MoE experts in a Qwen3 MoE litgpt checkpoint.

    Each layer's new safe experts are initialized by randomly copying from the
    existing *regular* experts in that layer.

    Arguments:
        checkpoint_dir: Path to the original litgpt checkpoint directory
            (must contain ``lit_model.pth`` and ``model_config.yaml``).
        num_new_safe_experts: How many *additional* safe experts to add per MoE
            layer.  If the model currently has ``n_safe_expert=0``, this creates
            all safe experts from scratch.
        seed: Random seed for reproducibility.
        output_dir: Where to write the expanded checkpoint.  Defaults to
            ``<checkpoint_dir>-Ext-<num_new_safe_experts>-Seed-<seed>`` in the
            same parent directory.
        dtype: Optional dtype conversion (e.g. ``"bfloat16"``).
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)

    if output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}-Ext-{num_new_safe_experts}-Seed-{seed}"

    pprint(locals())

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    if config.mlp_class_name != "LLaMAMoE":
        raise ValueError(
            f"This script only supports MoE models (mlp_class_name='LLaMAMoE'), "
            f"but the config has mlp_class_name={config.mlp_class_name!r}."
        )

    old_n_safe = config.n_safe_expert
    new_n_safe = old_n_safe + num_new_safe_experts
    n_expert = config.n_expert
    print(f"Regular experts (unchanged): {n_expert}")
    print(f"Expanding safe experts: {old_n_safe} -> {new_n_safe} (+{num_new_safe_experts} per MoE layer)")

    if dtype is not None:
        dtype = getattr(torch, dtype)

    # ------------------------------------------------------------------
    # Determine which layers are MoE (respecting first_k_dense_replace)
    # ------------------------------------------------------------------
    dense_layers: Set[int] = set()
    if config.first_k_dense_replace is not None:
        dense_layers = set(range(config.first_k_dense_replace))

    moe_layers: Set[int] = set(range(config.n_layer)) - dense_layers

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lit_model.pth"

    # Copy tokenizer and other ancillary files
    for fname in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "generation_config.json", "special_tokens_map.json"):
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # ------------------------------------------------------------------
    # Build the random expert mapping:
    #   layer -> list of source *regular* expert ids for each new safe expert
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed)
    layer_mapping: Dict[int, List[int]] = {}
    for layer_idx in sorted(moe_layers):
        layer_mapping[layer_idx] = rng.randint(0, n_expert, size=num_new_safe_experts).tolist()

    print("\nSafe-expert copy mapping (layer -> [source regular expert for each new safe expert]):")
    for layer_idx in sorted(layer_mapping):
        print(f"  Layer {layer_idx:3d}: {layer_mapping[layer_idx]}")

    # ------------------------------------------------------------------
    # Regex patterns
    # ------------------------------------------------------------------
    # Regular expert weights (source for copying)
    expert_re = re.compile(
        r"^transformer\.h\.(\d+)\.mlp\.experts\.(\d+)\.(fc_1|fc_2|proj)\.weight$"
    )
    # Regular gate weight (source for safe_gate rows)
    gate_re = re.compile(
        r"^transformer\.h\.(\d+)\.mlp\.gate\.weight$"
    )
    # Existing safe expert weights (if any)
    safe_expert_re = re.compile(
        r"^transformer\.h\.(\d+)\.mlp\.safe_experts\.(\d+)\.(fc_1|fc_2|proj)\.weight$"
    )
    # Existing safe gate weight (if any)
    safe_gate_re = re.compile(
        r"^transformer\.h\.(\d+)\.mlp\.safe_gate\.weight$"
    )

    # ------------------------------------------------------------------
    # Pass 1: buffer regular expert weights and gate weights (sources)
    #         Also buffer existing safe expert / safe_gate weights if present
    # ------------------------------------------------------------------
    print("\nPass 1: buffering expert weights ...")
    # expert_buf[layer][expert_idx][suffix] = tensor
    expert_buf: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    gate_buf: Dict[int, torch.Tensor] = {}
    # existing safe expert/gate buffers
    safe_expert_buf: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    safe_gate_buf: Dict[int, torch.Tensor] = {}

    lit_weights = lazy_load(checkpoint_dir / "lit_model.pth")
    lit_weights = lit_weights.get("model", lit_weights)

    for name, param in lit_weights.items():
        # Regular experts
        m = expert_re.match(name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx in moe_layers:
                expert_idx = int(m.group(2))
                suffix = m.group(3) + ".weight"
                expert_buf[layer_idx][expert_idx][suffix] = _load(param, dtype)
            continue

        # Regular gate
        mg = gate_re.match(name)
        if mg:
            layer_idx = int(mg.group(1))
            if layer_idx in moe_layers:
                gate_buf[layer_idx] = _load(param, dtype)
            continue

        # Existing safe experts
        ms = safe_expert_re.match(name)
        if ms:
            layer_idx = int(ms.group(1))
            if layer_idx in moe_layers:
                expert_idx = int(ms.group(2))
                suffix = ms.group(3) + ".weight"
                safe_expert_buf[layer_idx][expert_idx][suffix] = _load(param, dtype)
            continue

        # Existing safe gate
        msg = safe_gate_re.match(name)
        if msg:
            layer_idx = int(msg.group(1))
            if layer_idx in moe_layers:
                safe_gate_buf[layer_idx] = _load(param, dtype)
            continue

    print(f"  Buffered regular experts for {len(expert_buf)} layers")
    if safe_expert_buf:
        print(f"  Buffered existing safe experts for {len(safe_expert_buf)} layers")

    # ------------------------------------------------------------------
    # Pass 2: build expanded state dict with incremental_save
    # ------------------------------------------------------------------
    print("Pass 2: writing expanded checkpoint ...")
    sd: Dict[str, torch.Tensor] = {}

    lit_weights = lazy_load(checkpoint_dir / "lit_model.pth")
    lit_weights = lit_weights.get("model", lit_weights)

    # Track which layers we've already emitted safe_gate / safe_experts for
    emitted_safe_gate: Set[int] = set()
    emitted_safe_experts_suffix: Dict[int, Set[str]] = defaultdict(set)

    with incremental_save(output_path) as saver:
        for name, param in lit_weights.items():
            # --- Existing safe_gate: expand with new rows ---
            msg = safe_gate_re.match(name)
            if msg:
                layer_idx = int(msg.group(1))
                if layer_idx in moe_layers:
                    # Expand: keep existing rows + add new rows from regular gate
                    existing_safe_gate = safe_gate_buf[layer_idx]
                    mapping = layer_mapping[layer_idx]
                    regular_gate = gate_buf[layer_idx]
                    new_rows = regular_gate[mapping]  # (num_new_safe_experts, n_embd)
                    expanded = torch.cat([existing_safe_gate, new_rows], dim=0)
                    sd[name] = saver.store_early(expanded)
                    emitted_safe_gate.add(layer_idx)
                    continue
                else:
                    param = _load(param, dtype)
                    sd[name] = saver.store_early(param)
                    continue

            # --- Existing safe expert weights: pass through, emit new after last ---
            ms = safe_expert_re.match(name)
            if ms:
                layer_idx = int(ms.group(1))
                expert_idx = int(ms.group(2))
                suffix = ms.group(3) + ".weight"
                param = _load(param, dtype)
                sd[name] = saver.store_early(param)

                # After last existing safe expert of this suffix, emit new ones
                if expert_idx == old_n_safe - 1 and layer_idx in moe_layers:
                    mapping = layer_mapping[layer_idx]
                    for new_offset, src_idx in enumerate(mapping):
                        new_safe_idx = old_n_safe + new_offset
                        new_name = f"transformer.h.{layer_idx}.mlp.safe_experts.{new_safe_idx}.{suffix}"
                        src_tensor = expert_buf[layer_idx][src_idx][suffix]
                        sd[new_name] = saver.store_early(src_tensor.clone())
                    emitted_safe_experts_suffix[layer_idx].add(suffix)
                continue

            # --- Regular gate: pass through, but if no existing safe_gate, create one ---
            mg = gate_re.match(name)
            if mg:
                layer_idx = int(mg.group(1))
                param = _load(param, dtype)
                sd[name] = saver.store_early(param)

                # If original model had n_safe_expert=0, we need to create safe_gate
                if old_n_safe == 0 and layer_idx in moe_layers and layer_idx not in emitted_safe_gate:
                    mapping = layer_mapping[layer_idx]
                    regular_gate = gate_buf[layer_idx]
                    new_safe_gate = regular_gate[mapping]  # (num_new_safe_experts, n_embd)
                    safe_gate_name = f"transformer.h.{layer_idx}.mlp.safe_gate.weight"
                    sd[safe_gate_name] = saver.store_early(new_safe_gate)
                    emitted_safe_gate.add(layer_idx)
                continue

            # --- Regular expert weights: pass through, create safe_experts if needed ---
            m = expert_re.match(name)
            if m:
                layer_idx = int(m.group(1))
                expert_idx = int(m.group(2))
                suffix = m.group(3) + ".weight"
                param = _load(param, dtype)
                sd[name] = saver.store_early(param)

                # If original had n_safe_expert=0, emit all safe experts after last
                # regular expert of each suffix
                if (old_n_safe == 0
                        and expert_idx == n_expert - 1
                        and layer_idx in moe_layers
                        and suffix not in emitted_safe_experts_suffix[layer_idx]):
                    mapping = layer_mapping[layer_idx]
                    for new_offset, src_idx in enumerate(mapping):
                        new_name = f"transformer.h.{layer_idx}.mlp.safe_experts.{new_offset}.{suffix}"
                        src_tensor = expert_buf[layer_idx][src_idx][suffix]
                        sd[new_name] = saver.store_early(src_tensor.clone())
                    emitted_safe_experts_suffix[layer_idx].add(suffix)
                continue

            # --- All other weights: pass through ---
            param = _load(param, dtype)
            sd[name] = saver.store_early(param)

        saver.save(sd)

    # ------------------------------------------------------------------
    # Free buffers
    # ------------------------------------------------------------------
    del expert_buf, gate_buf, safe_expert_buf, safe_gate_buf, sd

    # ------------------------------------------------------------------
    # Write updated config
    # ------------------------------------------------------------------
    new_config = Config.from_file(checkpoint_dir / "model_config.yaml", n_safe_expert=new_n_safe)
    save_config(new_config, output_dir)

    print(f"\nDone! Expanded checkpoint saved to: {output_dir}")
    print(f"  n_expert (unchanged):    {n_expert}")
    print(f"  Old n_safe_expert:       {old_n_safe}")
    print(f"  New n_safe_expert:       {new_n_safe}")
    print(f"  n_expert_per_token (unchanged): {config.n_expert_per_token}")


def _load(param, dtype):
    """Materialise a lazy tensor and optionally cast dtype."""
    if hasattr(param, "_load_tensor"):
        param = param._load_tensor()
    if dtype is not None and param.dtype != dtype:
        param = param.to(dtype)
    return param


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(expand_moe_experts)
