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
OUTPUT_DATA_DIR = OUTPUT_DATASET_DIR
TRAIN_OUTPUT = OUTPUT_DATA_DIR / "train.parquet"
VALIDATION_OUTPUT = OUTPUT_DATA_DIR / "validation.parquet"
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
    raise FileNotFoundError(f"在 {base_dir} 或 {nested} 下未找到 parquet 文件")


def collect_split_files(base_dir: Path) -> Tuple[List[Path], List[Path]]:
    data_root = resolve_data_root(base_dir)
    train_files = sorted(data_root.glob("train-*.parquet"))
    val_files = sorted(data_root.glob("validation-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"在 {data_root} 下未找到 train-*.parquet")
    if not val_files:
        raise FileNotFoundError(f"在 {data_root} 下未找到 validation-*.parquet")
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
            print(f"检测到 GPU {rank} 的分片已完成，跳过加载模型。")
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

        print(f"GPU {rank} 开始翻译，条数: {len(texts)}")

        prompts = [
            f"<|im_start|>system\nYou are a professional translator. Translate to Spanish.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            for text in texts
        ]

        outputs = llm.generate(prompts, sampling_params)

        translated_texts = [output.outputs[0].text for output in outputs]
        shard_df = pd.DataFrame({"english": texts, "spanish": translated_texts})

        shard_df.to_parquet(shard_file)
        print(f"✅ GPU {rank} 翻译完成并已保存至断点文件。")

    except Exception as e:
        print(f"❌ 进程 {rank} 出错: {e}")


def main():
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_files, val_files = collect_split_files(SOURCE_DATASET_DIR)
    print(f"检测到 train 文件数: {len(train_files)}")
    print(f"检测到 validation 文件数: {len(val_files)}")

    print("开始将 train 全量加载到内存...")
    train_texts = load_texts(train_files)
    print("开始将 validation 加载到内存...")
    val_texts = load_texts(val_files)
    train_splits = split_evenly(train_texts, TRAIN_RANKS)

    print(f"输入数据集: {SOURCE_DATASET_DIR}")
    print(f"输出数据集: {OUTPUT_DATASET_DIR}")

    world_size = torch.cuda.device_count()
    if world_size != EXPECTED_GPUS:
        raise RuntimeError(f"需要 {EXPECTED_GPUS} 张 GPU（当前 {world_size} 张）")
    print(f"准备在 {world_size} 张 GPU 上执行任务（0-6=train, 7=validation）...")

    # 启动多进程
    mp.set_start_method("spawn", force=True)
    processes = []
    for i in range(world_size):
        shard_texts = train_splits[i] if i < TRAIN_RANKS else val_texts
        p = mp.Process(target=worker_task, args=(i, MODEL_DIR, shard_texts))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 最后合并结果（train / validation 分开输出）
    print("检查所有分片并准备按 train/validation 合并...")
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
        print(f"⚠️ 警告: 缺少分片 {missing_ranks}，未执行最终合并。")
        print("请修复后重新运行脚本，脚本会从断点处继续。")
        return

    train_df = pd.concat(train_shards, ignore_index=True)
    train_df.to_parquet(TRAIN_OUTPUT)
    validation_shard.to_parquet(VALIDATION_OUTPUT)
    print(f"🎉 合并完成！train 保存至: {TRAIN_OUTPUT}")
    print(f"🎉 合并完成！validation 保存至: {VALIDATION_OUTPUT}")


if __name__ == "__main__":
    main()
