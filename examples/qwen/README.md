# Qwen3 MoE Pre-training Examples

## Qwen3-30B-A3B

This directory contains pre-training scripts for the Qwen3-30B-A3B MoE model
using Megatron-LM on A100 GPUs.

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Total / Activated params | ~30B / ~3B |
| Hidden size | 2048 |
| Layers | 48 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 4 |
| Head dimension | 128 |
| Experts | 128 (top-8 routing) |
| Expert FFN hidden | 768 |
| Shared expert FFN | 6144 |
| Vocab size | 151936 |
| Max sequence length | 40960 |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Position embedding | RoPE (theta=1M) |
| Routing | Sigmoid + norm_topk_prob |

### Files

- `train_qwen3_30b_a3b_a100.sh` — Main training script for `torchrun` launch
- `slurm_train_qwen3_30b_a3b_a100.sbatch` — Slurm batch script for multi-node clusters

### Quick Start (mock data test)

```bash
cd /path/to/Megatron-LM
bash examples/qwen/train_qwen3_30b_a3b_a100.sh /tmp/test_ckpt MOCK MOCK
```

### Single Node (8× A100-80GB)

```bash
bash examples/qwen/train_qwen3_30b_a3b_a100.sh \
    /path/to/checkpoints \
    /path/to/qwen3-tokenizer \
    /path/to/data_prefix_text_document
```

### Multi-Node via Slurm

```bash
# Edit slurm script to set your paths, then:
sbatch examples/qwen/slurm_train_qwen3_30b_a3b_a100.sbatch

# Or override on command line:
CHECKPOINT_PATH=/path/to/ckpt TOKENIZER_PATH=/path/to/tokenizer \
DATA_PATH=/path/to/data sbatch examples/qwen/slurm_train_qwen3_30b_a3b_a100.sbatch
```

### Multi-Node (manual torchrun)

```bash
# Node 0:
MASTER_ADDR=<node0_ip> NNODES=4 NODE_RANK=0 \
    bash examples/qwen/train_qwen3_30b_a3b_a100.sh /path/ckpt /path/tok /path/data

# Node 1..3:
MASTER_ADDR=<node0_ip> NNODES=4 NODE_RANK=<1|2|3> \
    bash examples/qwen/train_qwen3_30b_a3b_a100.sh /path/ckpt /path/tok /path/data
```

### Parallelism Configuration

| GPUs | EP | TP | PP | DP | GPUs/Expert |
|------|----|----|----|----|-------------|
| 8 (1 node) | 8 | 1 | 1 | 1 | 1 |
| 16 (2 nodes) | 8 | 1 | 1 | 2 | 2 |
| 32 (4 nodes) | 8 | 1 | 1 | 4 | 4 |

Override via environment: `EXPERT_PARALLEL_SIZE=16 bash ...`

### Data Preparation

Data should be preprocessed into Megatron's binary format using
`tools/preprocess_data.py`. The Qwen3 tokenizer files (tokenizer.json,
vocab.json, etc.) should be provided as the tokenizer path.

### Notes

- `--moe-router-score-function sigmoid` is used to match Qwen3's sigmoid routing
  (vs. the default softmax). Megatron may not yet fully support `norm_topk_prob`
  — verify loss convergence against a HuggingFace reference.
- `--qk-layernorm` enables Q/K RMSNorm in attention (Qwen3-specific).
- `--kv-channels 128` ensures head_dim=128 (Qwen3 explicitly sets this;
  the auto-derived value would be 2048/32=64).
- FP8 training support is available if you add `--fp8-format hybrid` and related
  FP8 arguments (see `examples/llama/` for reference).
