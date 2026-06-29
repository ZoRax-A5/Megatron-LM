#!/bin/bash
#
# Pre-training script for Qwen3-30B-A3B (MoE) on A100 GPUs using Megatron-LM.
#
# Qwen3-30B-A3B architecture (from config.json):
#   - Model type: Qwen3 MoE (qwen3_moe)
#   - Total parameters: ~30B, Activated parameters: ~3B
#   - hidden_size: 2048, num_layers: 48
#   - num_attention_heads: 32, num_key_value_heads: 4 (GQA), head_dim: 128
#   - num_experts: 128, top-k: 8, moe_intermediate_size: 768
#   - shared_expert_intermediate_size: 6144 (with sigmoid gate)
#   - vocab_size: 151936, max_position_embeddings: 40960
#   - RMSNorm, SwiGLU, RoPE (theta=1M), QK LayerNorm
#   - sigmoid routing with norm_topk_prob
#
# Usage:
#   bash examples/qwen/train_qwen3_30b_a3b_a100.sh \
#       [checkpoint_path] [tokenizer_path] [data_path]
#
# Defaults:
#   checkpoint_path : /mnt/si002365wekc/zwx/checkpoints/qwen
#   tokenizer_path  : /mnt/si002365wekc/zwx/models/Qwen3-30B-A3B-tokenizer
#   data_path       : /mnt/si002365wekc/zwx/datasets/wikitext-103/wikitext_qwen3_text_document
#
# Example (with defaults):
#   bash examples/qwen/train_qwen3_30b_a3b_a100.sh
#
# Example (custom paths):
#   bash examples/qwen/train_qwen3_30b_a3b_a100.sh \
#       /path/to/checkpoints \
#       /path/to/qwen3-tokenizer \
#       /path/to/data_prefix_text_document
#
# For multi-node training, set the following environment variables:
#   MASTER_ADDR, MASTER_PORT, NNODES, NODE_RANK
#
# For quick testing with mock data:
#   bash examples/qwen/train_qwen3_30b_a3b_a100.sh /tmp/test_ckpt MOCK MOCK

set -e

###############################################################################
# Environment
###############################################################################
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}  # Adjust for your cluster
# export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

###############################################################################
# Paths (positional arguments)
###############################################################################
DEFAULT_CHECKPOINT_PATH="/mnt/si002365wekc/zwx/checkpoints/qwen"
DEFAULT_TOKENIZER_PATH="/mnt/si002365wekc/zwx/models/Qwen3-30B-A3B-tokenizer"
DEFAULT_DATA_PATH="/mnt/si002365wekc/zwx/datasets/wikitext-103/wikitext_qwen3_text_document"

CHECKPOINT_PATH=${1:-$DEFAULT_CHECKPOINT_PATH}
TOKENIZER_PATH=${2:-$DEFAULT_TOKENIZER_PATH}
DATA_PATH=${3:-$DEFAULT_DATA_PATH}

TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-"${CHECKPOINT_PATH}/tensorboard"}

# Create directories
mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"

DATA_CACHE_PATH="${DATA_CACHE_PATH:-${CHECKPOINT_PATH}/data_cache}"
mkdir -p "$DATA_CACHE_PATH"

###############################################################################
# Distributed training configuration
###############################################################################
# Single node defaults — override for multi-node via environment
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "=================================================="
echo " Qwen3-30B-A3B Pre-training on A100"
echo "=================================================="
echo " GPUS_PER_NODE : $GPUS_PER_NODE"
echo " NNODES        : $NNODES"
echo " NODE_RANK     : $NODE_RANK"
echo " WORLD_SIZE    : $WORLD_SIZE"
echo " MASTER_ADDR   : $MASTER_ADDR"
echo " MASTER_PORT   : $MASTER_PORT"
echo " Checkpoint    : $CHECKPOINT_PATH"
echo " Tokenizer     : $TOKENIZER_PATH"
echo " Data          : $DATA_PATH"
echo "=================================================="

###############################################################################
# Script path — assumes we are running from the Megatron-LM root
###############################################################################
PRETRAIN_SCRIPT="$(dirname "$0")/../../pretrain_gpt.py"
if [ ! -f "$PRETRAIN_SCRIPT" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT"
    echo "Please run this script from the Megatron-LM root directory."
    exit 1
fi

###############################################################################
# Model architecture (Qwen3-30B-A3B)
###############################################################################
# Notes:
#   - kv_channels=128 matches Qwen3's head_dim (hidden_size / num_heads would
#     give 64, but Qwen3 explicitly uses head_dim=128).
#   - QK layernorm is enabled (--qk-layernorm); Qwen3 applies RMSNorm to Q and K
#     before computing attention scores.
#   - init-method-std=0.02 from the model config (initializer_range).

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 40960
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --init-method-std 0.02
    --attention-backend flash
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --qk-layernorm
)

###############################################################################
# MoE arguments
###############################################################################
# Qwen3-30B-A3B uses:
#   - 128 experts with top-8 routing (sigmoid scoring + norm_topk_prob)
#   - Per-expert FFN hidden size: 768
#   - Shared experts with intermediate size: 6144 and a sigmoid gate
#   - MoE every layer (decoder_sparse_step=1)
#   - Router auxiliary loss coefficient: 0.001

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-ffn-hidden-size 768
    --moe-shared-expert-intermediate-size 6144
    --moe-shared-expert-gate
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq 1
)

###############################################################################
# Training arguments
###############################################################################
# BF16 training with distributed optimizer.
# Adjust --global-batch-size based on the number of GPUs.
#   - 8 GPUs (1 node): global_batch_size = 1024 (128 per GPU)
#   - 16 GPUs (2 nodes): global_batch_size = 2048

DEFAULT_GLOBAL_BATCH_SIZE=1024
if [ "$WORLD_SIZE" -gt 8 ]; then
    DEFAULT_GLOBAL_BATCH_SIZE=$((128 * WORLD_SIZE))
fi
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$DEFAULT_GLOBAL_BATCH_SIZE}

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 500000
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --lr-decay-iters 480000
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
    --exit-duration-in-mins 230
)

###############################################################################
# Model parallelism
###############################################################################
# For A100-80GB x 8 GPUs (1 node):
#   - TP=1, PP=1, EP=8
#   - Each GPU holds 128/8 = 16 experts
#   - Memory ~45-50GB per GPU, fits in A100-80GB
#
# For multi-node scaling, adjust EP size (must divide num_experts=128):
#   - 16 GPUs: EP=8, DP=2  (or EP=16, DP=1)
#   - 32 GPUs: EP=8, DP=4  (or EP=16, DP=2)

EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size $EXPERT_PARALLEL_SIZE
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

###############################################################################
# Data / tokenizer
###############################################################################
if [[ "$TOKENIZER_PATH" == "MOCK" ]] || [[ "$DATA_PATH" == "MOCK" ]]; then
    echo ">>> Using MOCK data (no real data or tokenizer)"
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 151936
        --data-cache-path "$DATA_CACHE_PATH"
        --tiktoken-pattern v2
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    echo ">>> Using real data and tokenizer"
    DATA_ARGS=(
        --data-path "$DATA_PATH"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_PATH"
        --vocab-size 151936
        --data-cache-path "$DATA_CACHE_PATH"
        --split "99,1,0"
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
fi

###############################################################################
# Evaluation and logging
###############################################################################
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --eval-interval 100
    --eval-iters 32
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --no-load-optim
    --no-load-rng
)

###############################################################################
# Launch training with torchrun
###############################################################################
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

echo ""
echo "Launching training..."
echo "Command: torchrun ${DISTRIBUTED_ARGS[*]} ${PRETRAIN_SCRIPT} ..."
echo ""

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$PRETRAIN_SCRIPT" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"
