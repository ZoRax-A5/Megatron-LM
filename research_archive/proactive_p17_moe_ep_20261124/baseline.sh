#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
EP=${1:?EP size}
shift
torchrun --standalone --nproc_per_node=8 "${P17_ENTRY:-pretrain_gpt.py}" --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size "$EP" --num-layers 6 --hidden-size 512 --ffn-hidden-size 2048 --num-attention-heads 8 --seq-length 2048 --max-position-embeddings 2048 --micro-batch-size 1 --global-batch-size 32 --train-iters 50 --eval-interval 1000 --eval-iters 0 --lr 0.0001 --min-lr 0.00001 --lr-decay-style cosine --log-interval 1 --num-experts 8 --moe-router-topk 2 --moe-token-dispatcher-type alltoall --moe-grouped-gemm --disable-bias-linear --mock-data --tokenizer-type NullTokenizer --vocab-size 1024 --bf16 --seed 42 "$@"
