#!/bin/bash
# Evaluate MMLU on original and expanded Qwen3-30B-A3B models
# Usage: bash scripts/eval_mmlu.sh

set -e

BATCH_SIZE=${BATCH_SIZE:-4}
TASKS=${TASKS:-"mmlu"}
LIMIT=${LIMIT:-""}  # set e.g. LIMIT=10 for quick testing
OFFLINE=${OFFLINE:-1}  # set OFFLINE=0 to allow online fetch

SHARED="/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe"

# Networking/cache defaults for environments where huggingface.co is unstable.
HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
HF_HOME=${HF_HOME:-"$SHARED/.cache/huggingface"}

# Set DISABLE_PROXY=1 to force direct connection (useful when 127.0.0.1 proxy is not available on remote machines).
if [ "${DISABLE_PROXY:-0}" = "1" ]; then
    clearproxy
fi

mkdir -p "$HF_HOME"
export HF_ENDPOINT HF_HOME

if [ "$OFFLINE" = "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export DATASETS_OFFLINE=1
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "OFFLINE=$OFFLINE"
if [ "$OFFLINE" = "1" ]; then
    echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
    echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
    echo "DATASETS_OFFLINE=$DATASETS_OFFLINE"
fi

ORIG_CKPT="$SHARED/checkpoints/Qwen3-30B-A3B-Base"
EXT_CKPT="$SHARED/checkpoints/Qwen3-30B-A3B-Base-Ext-8-Seed-42"

ORIG_OUT="$SHARED/evaluate_model/Qwen3-30B-A3B-Base"
EXT_OUT="$SHARED/evaluate_model/Qwen3-30B-A3B-Base-Ext-8-Seed-42"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "============================================="
echo "  Evaluating ORIGINAL model: $ORIG_CKPT"
echo "============================================="
litgpt evaluate "$ORIG_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --tasks "$TASKS" \
    --out_dir "$ORIG_OUT" \
    $LIMIT_ARG

echo ""
echo "============================================="
echo "  Evaluating EXPANDED model: $EXT_CKPT"
echo "============================================="
litgpt evaluate "$EXT_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --tasks "$TASKS" \
    --out_dir "$EXT_OUT" \
    $LIMIT_ARG

echo ""
echo "============================================="
echo "  Results summary"
echo "============================================="
echo "Original model results: $ORIG_OUT/results.json"
echo "Expanded model results: $EXT_OUT/results.json"
echo ""

# Print side-by-side comparison if both results exist
python3 -c "
import json, sys

orig_path = '$ORIG_OUT/results.json'
ext_path  = '$EXT_OUT/results.json'

try:
    with open(orig_path) as f:
        orig = json.load(f)
    with open(ext_path) as f:
        ext = json.load(f)
except FileNotFoundError as e:
    print(f'Could not load results: {e}')
    sys.exit(0)

print(f'{\"Task\":<40} {\"Original\":>10} {\"Expanded\":>10} {\"Diff\":>10}')
print('-' * 72)
for task in sorted(set(list(orig.get('results', {}).keys()) + list(ext.get('results', {}).keys()))):
    orig_acc = orig.get('results', {}).get(task, {}).get('acc,none', None)
    ext_acc  = ext.get('results', {}).get(task, {}).get('acc,none', None)
    if orig_acc is not None and ext_acc is not None:
        diff = ext_acc - orig_acc
        print(f'{task:<40} {orig_acc:>10.4f} {ext_acc:>10.4f} {diff:>+10.4f}')
    elif orig_acc is not None:
        print(f'{task:<40} {orig_acc:>10.4f} {\"N/A\":>10}')
    elif ext_acc is not None:
        print(f'{task:<40} {\"N/A\":>10} {ext_acc:>10.4f}')
"
