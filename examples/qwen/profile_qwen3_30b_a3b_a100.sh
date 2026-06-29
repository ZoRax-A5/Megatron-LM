#!/bin/bash
# Profile Qwen3-30B-A3B GPU memory on 8 nodes x 8 A100 GPUs by default.

set -e

CHECKPOINT_PATH=${1:?"Error: checkpoint path is required"}
TOKENIZER_PATH=${2:?"Error: tokenizer path is required"}
DATA_PATH=${3:?"Error: data path is required"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE_DIR=${MEMORY_PROFILE_DIR:-"${CHECKPOINT_PATH}/memory_profile"}
MEMORY_PROFILE_MODE=${MEMORY_PROFILE_MODE:-deep}
MEMORY_PROFILE_WARMUP_ITERS=${MEMORY_PROFILE_WARMUP_ITERS:-5}
MEMORY_PROFILE_ITERS=${MEMORY_PROFILE_ITERS:-3}

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-8}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
export PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-8}
export EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export TRAIN_ITERS=${TRAIN_ITERS:-500000}

mkdir -p "${PROFILE_DIR}"

bash "${SCRIPT_DIR}/train_qwen3_30b_a3b_a100.sh" \
    "${CHECKPOINT_PATH}" "${TOKENIZER_PATH}" "${DATA_PATH}" \
    --memory-profile-mode "${MEMORY_PROFILE_MODE}" \
    --memory-profile-warmup-iters "${MEMORY_PROFILE_WARMUP_ITERS}" \
    --memory-profile-iters "${MEMORY_PROFILE_ITERS}" \
    --memory-profile-dir "${PROFILE_DIR}" \
    --no-save-optim \
    --no-save-rng \
    "${@:4}"

if [[ "${NODE_RANK:-0}" == "0" ]]; then
    python "${SCRIPT_DIR}/../../tools/memory_profile/plot_memory_profile.py" \
        --input-dir "${PROFILE_DIR}" \
        --expected-ranks "$((GPUS_PER_NODE * NNODES))"
    echo "Memory profile report: ${PROFILE_DIR}/plots/index.html"
fi
