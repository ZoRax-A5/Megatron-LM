"""Temporary per-run Qwen driver that preserves checkpoint lifecycle events."""

import json
import os
import runpy
import sys
from pathlib import Path

import torch

bench_root = Path(__file__).resolve().parents[2]
repo = Path(os.environ["QWEN_BENCH_MEGATRON"])
case_dir = Path(os.environ["QWEN_BENCH_CASE"])
sys.path.insert(0, str(bench_root / "qwen4b_ab"))
sys.path.insert(0, str(repo))
import load_weights  # noqa: F401

from megatron.core.tensor_parallel.random import ProactiveCheckpoint
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

records = []
forward_calls = {}
original_cleanup = ProactiveCheckpoint._cleanup_after_backward
original_forward = TELayerNormColumnParallelLinear.forward


def capture_cleanup(handle):
    original_cleanup(handle)
    records.append(
        {
            "checkpoint_id": handle.checkpoint_id,
            "metadata": handle.metadata,
            "events": list(handle.events),
            "cache_lifetime_seconds": handle.cache_lifetime_seconds,
            "memory_snapshot": dict(handle.memory_snapshot),
            "final_state": handle.state,
        }
    )


def capture_forward(module, *args, **kwargs):
    tag = getattr(module, "_p1tag", None)
    key = str(tag)
    forward_calls[key] = forward_calls.get(key, 0) + 1
    return original_forward(module, *args, **kwargs)


ProactiveCheckpoint._cleanup_after_backward = capture_cleanup
TELayerNormColumnParallelLinear.forward = capture_forward
delegate = Path(__file__).resolve().parents[2] / "qwen4b_c" / (
    "gradient_entry.py" if os.environ.get("C_KIND") == "verify" else "performance_entry.py"
)
try:
    runpy.run_path(str(delegate), run_name="__main__")
finally:
    torch.cuda.synchronize()
    (case_dir / f"mechanism_rank{os.environ['RANK']}.json").write_text(
        json.dumps(
            {
                "rank": int(os.environ["RANK"]),
                "forward_calls_by_tag": forward_calls,
                "checkpoints": records,
            },
            indent=2,
        )
    )
