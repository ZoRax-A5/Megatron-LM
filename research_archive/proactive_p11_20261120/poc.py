"""Capture a warm training step of the P9 schedule or the SP compatibility control."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

here = Path(__file__).resolve().parent
root = here.parents[1]
repo = root.parent / "Megatron-LM-layerwise"
sys.path.insert(0, str(root))
import bench

parser = argparse.ArgumentParser()
parser.add_argument("size", choices=("small","4b"))
parser.add_argument("mode", choices=("S0","C1early","C1late","same"))
parser.add_argument("measurement", choices=("trace","verify","perf"))
parser.add_argument("--connections", default="1")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--suffix", default="")
opts = parser.parse_args()
mode = opts.mode
large = opts.size == "4b"
ab = root / "results/qwen4b_ab_20261118"
config = (json.loads((ab / "model_config.json").read_text()) if large else
          dict(num_hidden_layers=2, hidden_size=256, intermediate_size=512,
               num_attention_heads=4, num_key_value_heads=2, head_dim=64,
               vocab_size=2048, rms_norm_eps=1e-6, rope_theta=1000000,
               initializer_range=0.02, max_position_embeddings=512))
case = here / ("p11_" + "_".join((opts.size,mode,opts.measurement,"conn"+opts.connections)) + opts.suffix)
case.mkdir()
cmd = bench.command(SimpleNamespace(gpus=8, megatron=repo, seed=1234, iters=opts.steps,
    seq_length=2048 if large else 256, micro_batch=1, global_batch=32,
    sequence_parallel=False, overlap=False), config, 2, 1)
cmd[cmd.index(str(repo / "pretrain_gpt.py"))] = str(here / "poc_entry.py")
cmd += ["--overlap-grad-reduce", "--ddp-bucket-size", "40000000" if large else "400000"]
mapping = {"0": ["attn_qkv"], "1": ["attn_qkv"]}
if large:
    mapping["17"] = ["attn_qkv"]
cmd += ["--recompute-granularity", "selective", "--recompute-layer-config", json.dumps(mapping)]
if mode == "same":
    cmd += ["--cross-microbatch-recompute-schedule"]
if mode != "S0":
    cap = len(mapping) if mode == "same" else 1
    cmd += ["--proactive-recompute", "--proactive-recompute-max-cached-checkpoints", str(cap)]
if mode.startswith("C1"):
    cmd += ["--proactive-recompute-dp-bucket", "0" if mode=="C1early" else "35"]
settings = dict(QWEN_BENCH_MEGATRON=str(repo), QWEN_BENCH_CASE=str(case),
                P9_PHASE="timeline" if opts.measurement=="trace" else "perf", P10_CAPTURE_STEP="2",
                CUDA_DEVICE_MAX_CONNECTIONS=opts.connections, P10_POC=mode,
                OMP_NUM_THREADS="2", NCCL_DEBUG="WARN", NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
                CUBLAS_WORKSPACE_CONFIG=":4096:8")
if opts.measurement == "verify":
    settings["P10_EXACT"] = "1"
if large:
    settings["AB_CONVERTED"] = str(ab / "converted")
with tempfile.TemporaryDirectory(prefix="p10_nsys_") as tmp:
    if opts.measurement == "trace":
        cmd = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=none",
               "--wait=all", "--export=sqlite", "-o", str(Path(tmp)/"trace"), *cmd]
    (case / "manifest.json").write_text(json.dumps(dict(cmd=cmd, env=settings), indent=2))
    with (case / "train.log").open("w") as log:
        result = subprocess.run(cmd, cwd=repo, env={**os.environ, **settings},
                                stdout=log, stderr=subprocess.STDOUT, timeout=450)
    for path in Path(tmp).glob("trace.*"):
        shutil.move(path, case / path.name)
    (case/"status.json").write_text(json.dumps(dict(returncode=result.returncode)))
    result.check_returncode()
    if opts.measurement == "trace":
        assert (case / "trace.sqlite").exists()
print(f"COMPLETE {case}", flush=True)
