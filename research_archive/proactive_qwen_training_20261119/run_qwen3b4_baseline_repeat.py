"""Repeat the three-step no-recompute full-model baseline for reproducibility."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
results = Path(__file__).resolve().parent / "qwen3b4_seq2048_mbs1"
case = results / "no_recompute_repeat"
case.mkdir(exist_ok=False)
repo = Path("/mnt/si002365wekc/zwx/code/recom/Megatron-LM-layerwise")
ab = root / "results/qwen4b_ab_20261118"
sys.path.insert(0, str(root))
import bench

cfg = json.loads((ab / "model_config.json").read_text())
args = SimpleNamespace(
    gpus=8,
    megatron=repo,
    seed=1234,
    iters=3,
    seq_length=2048,
    micro_batch=1,
    global_batch=32,
    sequence_parallel=False,
    overlap=False,
)
command = bench.command(args, cfg, 2, 1)
command[command.index(str(repo / "pretrain_gpt.py"))] = str(
    Path(__file__).resolve().parent / "proactive_event_entry.py"
)
command[command.index("--standalone"):command.index("--standalone")] = [
    "--log-dir",
    str(case / "rank_logs"),
    "--redirects",
    "3",
    "--tee",
    "3",
]
env = os.environ.copy()
env.update(
    {
        "QWEN_BENCH_MEGATRON": str(repo),
        "QWEN_BENCH_CASE": str(case),
        "P0_OUTPUT": str(case),
        "C_KIND": "verify",
        "AB_CONVERTED": str(ab / "converted"),
        "AB_VERIFY_LOAD": "1",
        "AB_WARMUP": "0",
        "P1_STRATEGY": "B1",
        "P1_REFERENCE": str(case),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "OMP_NUM_THREADS": "4",
        "NCCL_DEBUG": "WARN",
        "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
)
manifest = {
    "name": "qwen3-4b-no-recompute-baseline-repeat",
    "steps": 3,
    "same_config_as": str(results / "no_recompute/manifest.json"),
    "command": command,
}
(case / "manifest.json").write_text(json.dumps(manifest, indent=2))
started = time.time()
with (case / "train.log").open("w") as log:
    result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
(case / "run_status.json").write_text(
    json.dumps({"returncode": result.returncode, "seconds": time.time() - started}, indent=2)
)
if result.returncode:
    raise SystemExit(f"baseline repeat failed; inspect {case / 'train.log'}")
