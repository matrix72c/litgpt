import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams

# ===========================================
MODEL_DIR = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/3ffd1f50b179e643d839c86df9ffbbefcb0d5018"
SOURCE_DATASET_DIR = Path("data/tinystories")
OUTPUT_DATASET_DIR = Path("data/multilingual-tinystories")
CHECKPOINT_DIR = OUTPUT_DATASET_DIR / "checkpoints"
EN_DIR = OUTPUT_DATASET_DIR / "en"
ES_DIR = OUTPUT_DATASET_DIR / "es"
EN_TRAIN_OUTPUT = EN_DIR / "train.parquet"
EN_VALIDATION_OUTPUT = EN_DIR / "validation.parquet"
ES_TRAIN_OUTPUT = ES_DIR / "train.parquet"
ES_VALIDATION_OUTPUT = ES_DIR / "validation.parquet"
GPU_MEMORY_UTIL = 0.90
MAX_MODEL_LEN = 2048
# ==========================================

TRAIN_RANKS = 7
EXPECTED_GPUS = 8


def resolve_data_root(base_dir: Path) -> Path:
    nested = base_dir / "data"
    if nested.exists() and any(nested.glob("*.parquet")):
        return nested
    if any(base_dir.glob("*.parquet")):
        return base_dir
    raise FileNotFoundError(f"No parquet files found in {base_dir} or {nested}")


def collect_split_files(base_dir: Path) -> Tuple[List[Path], List[Path]]:
    data_root = resolve_data_root(base_dir)
    train_files = sorted(data_root.glob("train-*.parquet"))
    val_files = sorted(data_root.glob("validation-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train-*.parquet files found in {data_root}")
    if not val_files:
        raise FileNotFoundError(f"No validation-*.parquet files found in {data_root}")
    return train_files, val_files


def load_texts(files: List[Path]) -> List[str]:
    frames = [pd.read_parquet(file, columns=["text"]) for file in files]
    merged = pd.concat(frames, ignore_index=True)
    return merged["text"].astype(str).tolist()


def split_evenly(items: List[str], n_parts: int) -> List[List[str]]:
    base_size, remainder = divmod(len(items), n_parts)
    splits: List[List[str]] = []
    start = 0
    for i in range(n_parts):
        end = start + base_size + (1 if i < remainder else 0)
        splits.append(items[start:end])
        start = end
    return splits


def worker_task(rank, model_path, texts):
    try:
        shard_file = os.path.join(CHECKPOINT_DIR, f"shard_{rank}.parquet")

        if os.path.exists(shard_file):
            print(f"Shard for GPU {rank} already completed, skipping model loading.")
            return

        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=True,
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=512,
        )

        print(f"GPU {rank} starting translation, num texts: {len(texts)}")

        prompts = [
            f"<|im_start|>system\nYou are a professional translator. Translate to Spanish.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            for text in texts
        ]

        outputs = llm.generate(prompts, sampling_params)

        translated_texts = [output.outputs[0].text for output in outputs]
        shard_df = pd.DataFrame({"english": texts, "spanish": translated_texts})

        shard_df.to_parquet(shard_file)
        print(f"✅ GPU {rank} translation complete, checkpoint saved.")

    except Exception as e:
        print(f"❌ Process {rank} error: {e}")


def main():
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ES_DIR.mkdir(parents=True, exist_ok=True)

    train_files, val_files = collect_split_files(SOURCE_DATASET_DIR)
    print(f"Detected train files: {len(train_files)}")
    print(f"Detected validation files: {len(val_files)}")

    print("Loading all train data into memory...")
    train_texts = load_texts(train_files)
    print("Loading validation data into memory...")
    val_texts = load_texts(val_files)
    train_splits = split_evenly(train_texts, TRAIN_RANKS)

    print(f"Input dataset: {SOURCE_DATASET_DIR}")
    print(f"Output dataset: {OUTPUT_DATASET_DIR}")

    world_size = torch.cuda.device_count()
    if world_size != EXPECTED_GPUS:
        raise RuntimeError(f"Expected {EXPECTED_GPUS} GPUs (found {world_size})")
    print(f"Launching tasks on {world_size} GPUs (0-6=train, 7=validation)...")

    # Start multiprocessing
    mp.set_start_method("spawn", force=True)
    processes = []
    for i in range(world_size):
        shard_texts = train_splits[i] if i < TRAIN_RANKS else val_texts
        p = mp.Process(target=worker_task, args=(i, MODEL_DIR, shard_texts))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results (train / validation output separately)
    print("Checking all shards and preparing to merge by train/validation...")
    train_shards = []
    validation_shard = None
    missing_ranks = []

    for i in range(world_size):
        shard_file = os.path.join(CHECKPOINT_DIR, f"shard_{i}.parquet")
        if not os.path.exists(shard_file):
            missing_ranks.append(i)
            continue
        shard_df = pd.read_parquet(shard_file)
        if i < TRAIN_RANKS:
            train_shards.append(shard_df)
        else:
            validation_shard = shard_df

    if missing_ranks:
        print(f"⚠️ Warning: missing shards {missing_ranks}, final merge not performed.")
        print("Please fix the issue and rerun the script; it will resume from checkpoints.")
        return

    train_df = pd.concat(train_shards, ignore_index=True)

    # Save en and es separately
    train_df[["english"]].rename(columns={"english": "text"}).to_parquet(EN_TRAIN_OUTPUT)
    train_df[["spanish"]].rename(columns={"spanish": "text"}).to_parquet(ES_TRAIN_OUTPUT)
    validation_shard[["english"]].rename(columns={"english": "text"}).to_parquet(EN_VALIDATION_OUTPUT)
    validation_shard[["spanish"]].rename(columns={"spanish": "text"}).to_parquet(ES_VALIDATION_OUTPUT)

    print(f"🎉 Merge complete! English saved to: {EN_DIR}")
    print(f"🎉 Merge complete! Spanish saved to: {ES_DIR}")


if __name__ == "__main__":
    main()
