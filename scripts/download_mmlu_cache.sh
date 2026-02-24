#!/bin/bash
# Download/refresh lm_eval cache on an online machine without GPU.
# Usage: bash scripts/download_mmlu_cache.sh

set -e

TASKS=${TASKS:-"mmlu"}
LIMIT=${LIMIT:-1}
HF_TEST_MODEL=${HF_TEST_MODEL:-"hf-internal-testing/tiny-random-gpt2"}

SHARED="/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe"
HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
HF_HOME=${HF_HOME:-"$SHARED/.cache/huggingface"}

# Set DISABLE_PROXY=1 to force direct connection (useful when 127.0.0.1 proxy is not available on remote machines).
if [ "${DISABLE_PROXY:-0}" = "1" ]; then
    clearproxy
fi

mkdir -p "$HF_HOME"

# Force online mode for cache warming
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset DATASETS_OFFLINE
export HF_ENDPOINT HF_HOME TASKS LIMIT HF_TEST_MODEL

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "TASKS=$TASKS"
echo "LIMIT=$LIMIT"
echo "HF_TEST_MODEL=$HF_TEST_MODEL"

echo "============================================="
echo "  Warming lm_eval cache on CPU"
echo "============================================="
python3 - <<'PY'
import os
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

tasks = [t.strip() for t in os.environ["TASKS"].split(",") if t.strip()]
limit = float(os.environ["LIMIT"])
model_name = os.environ["HF_TEST_MODEL"]

print(f"Loading tiny test model on CPU: {model_name}")
model = HFLM(
    pretrained=model_name,
    device="cpu",
    batch_size=1,
    dtype="float32",
)

print(f"Downloading task assets for: {tasks}")
results = evaluator.simple_evaluate(
    model=model,
    tasks=tasks,
    batch_size=1,
    device="cpu",
    limit=limit,
    num_fewshot=0,
    random_seed=1234,
    numpy_random_seed=1234,
    torch_random_seed=1234,
)

print("Cache warmup finished. Summary keys:", list(results.keys()))
PY

echo ""
echo "Cache warmup complete."
echo "Now copy/sync this directory to offline GPU nodes if needed:"
echo "  $HF_HOME"
