"""Run matched three-step Qwen3-4B mechanism checks on the current worktree."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
results = Path(__file__).resolve().parent / "qwen3b4_seq2048_mbs1"
repo = Path("/mnt/si002365wekc/zwx/code/recom/Megatron-LM-layerwise")
baseline_reference = results / "no_recompute"
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
base_command = bench.command(args, cfg, 2, 1)
base_command[base_command.index(str(repo / "pretrain_gpt.py"))] = str(
    Path(__file__).resolve().parent / "proactive_event_entry.py"
)
mapping = {"0": ["attn_qkv"]}
runs = [
    ("no_recompute", "B1", False, False),
    ("on_demand_qkv", "QKV", True, False),
    ("proactive_qkv", "PROACTIVE", True, True),
]
results.mkdir(exist_ok=False)
summary = {"runs": [], "mapping": mapping, "command_base": base_command}
for name, strategy, with_mapping, proactive in runs:
    case = results / name
    case.mkdir()
    command = list(base_command)
    command[command.index("--train-iters") + 1] = "3"
    command[command.index("--seq-length") + 1] = "2048"
    command[command.index("--micro-batch-size") + 1] = "1"
    command[command.index("--global-batch-size") + 1] = "32"
    command[command.index("--standalone"):command.index("--standalone")] = [
        "--log-dir",
        str(case / "rank_logs"),
        "--redirects",
        "3",
        "--tee",
        "3",
    ]
    if with_mapping:
        command.extend(
            [
                "--recompute-granularity",
                "selective",
                "--recompute-layer-config",
                json.dumps(mapping, separators=(",", ":")),
            ]
        )
    if proactive:
        command.extend(
            ["--proactive-recompute", "--proactive-recompute-max-cached-checkpoints", "1"]
        )
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
            "P1_STRATEGY": strategy,
            "P1_REFERENCE": str(baseline_reference),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "OMP_NUM_THREADS": "4",
            "NCCL_DEBUG": "WARN",
            "PYTHONPATH": str(repo) + os.pathsep + env.get("PYTHONPATH", ""),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    with (case / "train.log").open("w") as log:
        started = time.time()
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    run_info = {
        "name": name,
        "strategy": strategy,
        "returncode": result.returncode,
        "seconds": time.time() - started,
        "command": command,
        "run_env": {
            key: env[key]
            for key in ("QWEN_BENCH_MEGATRON", "QWEN_BENCH_CASE", "C_KIND", "P1_STRATEGY", "P1_REFERENCE", "AB_VERIFY_LOAD")
        },
    }
    (case / "manifest.json").write_text(json.dumps(run_info, indent=2))
    summary["runs"].append(run_info)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    if result.returncode:
        raise SystemExit(f"{name} failed; see {case / 'train.log'}")
print(json.dumps(summary, indent=2))
