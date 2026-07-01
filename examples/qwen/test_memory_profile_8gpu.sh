#!/bin/bash
# Quick single-node 8-GPU smoke test for Qwen MoE memory profiling.
#
# This uses MOCK data and a reduced model size so the goal is to validate the
# profiler path, rank-local samples, report generation, and early exit.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CHECKPOINT_PATH=${1:-/tmp/megatron_qwen_memory_profile_8gpu}
TOKENIZER_PATH=${2:-MOCK}
DATA_PATH=${3:-MOCK}

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-6000}

export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
export PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
export EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
export TRAIN_ITERS=${TRAIN_ITERS:-20}

# eval_global_batch_size must be divisible by (eval_micro_batch_size * data_parallel_size).
# With TP=1, PP=1, DP defaults to GPUS_PER_NODE (8 here).
DATA_PARALLEL_SIZE=$(( GPUS_PER_NODE / TENSOR_PARALLEL_SIZE / PIPELINE_PARALLEL_SIZE ))
EVAL_GLOBAL_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE > DATA_PARALLEL_SIZE ? GLOBAL_BATCH_SIZE : DATA_PARALLEL_SIZE ))

MEMORY_PROFILE_DIR=${MEMORY_PROFILE_DIR:-"${CHECKPOINT_PATH}/memory_profile_smoke"}
MEMORY_PROFILE_MODE=${MEMORY_PROFILE_MODE:-light}
MEMORY_PROFILE_WARMUP_ITERS=${MEMORY_PROFILE_WARMUP_ITERS:-1}
MEMORY_PROFILE_ITERS=${MEMORY_PROFILE_ITERS:-1}

mkdir -p "${CHECKPOINT_PATH}" "${MEMORY_PROFILE_DIR}"

bash "${SCRIPT_DIR}/train_qwen3_30b_a3b_a100.sh" \
    "${CHECKPOINT_PATH}" "${TOKENIZER_PATH}" "${DATA_PATH}" \
    --num-layers 2 \
    --hidden-size 512 \
    --ffn-hidden-size 1024 \
    --num-attention-heads 8 \
    --num-query-groups 4 \
    --kv-channels 64 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 256 \
    --moe-shared-expert-intermediate-size 512 \
    --micro-batch-size 1 \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --train-iters "${TRAIN_ITERS}" \
    --lr-decay-iters 20 \
    --lr-warmup-iters 1 \
    --eval-iters 0 \
    --eval-global-batch-size "${EVAL_GLOBAL_BATCH_SIZE}" \
    --memory-profile-mode "${MEMORY_PROFILE_MODE}" \
    --memory-profile-warmup-iters "${MEMORY_PROFILE_WARMUP_ITERS}" \
    --memory-profile-iters "${MEMORY_PROFILE_ITERS}" \
    --memory-profile-dir "${MEMORY_PROFILE_DIR}" \
    --no-save-optim \
    --no-save-rng \
    "${@:4}"

python "${SCRIPT_DIR}/../../tools/memory_profile/plot_memory_profile.py" \
    --input-dir "${MEMORY_PROFILE_DIR}" \
    --expected-ranks "${GPUS_PER_NODE}"

echo "Memory profile smoke test report: ${MEMORY_PROFILE_DIR}/plots/index.html"
