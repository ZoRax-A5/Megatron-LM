"""Observe the normal pretrain entry point, with optional same-stream CUDA events."""
from collections import Counter
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import time

import torch

repo = Path(os.environ["QWEN_BENCH_MEGATRON"])
out = Path(os.environ["QWEN_BENCH_CASE"])
rank = int(os.environ["RANK"])
phase = os.environ["P9_PHASE"]
sys.path.insert(0, str(repo))
from megatron.core.pipeline_parallel import schedules
from megatron.core.tensor_parallel import random as recompute
from megatron.core.transformer import attention
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.distributed import param_and_grad_buffer
import megatron.training.training as training

steps = []
calls = Counter()
events = []
stage = None
current = None
active = None
original_train_step = training.train_step
original_qkv = TELayerNormColumnParallelLinear.forward
original_forward = schedules.forward_step
original_backward = schedules.backward_step
original_prefetch = recompute.ProactiveCheckpoint.prefetch
original_cleanup = recompute.ProactiveCheckpoint._cleanup_after_backward

def qkv(module, *args, **kwargs):
    if hasattr(module, "_p1tag") and module._p1tag[1] != "qkv":
        return original_qkv(module, *args, **kwargs)
    # Tag the fused QKV modules from SelfAttention; FC1 uses the same TE type.
    calls[str(stage)] += int(id(module) in qkv_modules)
    return original_qkv(module, *args, **kwargs)

qkv_modules = set()
original_get_qkv = attention.SelfAttention.get_query_key_value_tensors

def get_qkv(module, *args, **kwargs):
    qkv_modules.add(id(module.linear_qkv))
    return original_get_qkv(module, *args, **kwargs)

def mark(name):
    if phase == "timeline":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append((name, time.perf_counter(), event))

def forward(*args, **kwargs):
    global stage, current
    current = kwargs["current_microbatch"]
    stage = ("forward", current)
    mark(f"forward:{current}:start")
    if phase == "timeline":
        torch.cuda.nvtx.range_push(f"p9.forward.{current}")
    try:
        return original_forward(*args, **kwargs)
    finally:
        if phase == "timeline":
            torch.cuda.nvtx.range_pop()
        mark(f"forward:{current}:end")

def backward(*args, **kwargs):
    global stage
    index = active["backwards"]
    stage = ("backward", index)
    mark(f"backward:{index}:start")
    if phase == "timeline":
        torch.cuda.nvtx.range_push(f"p9.backward.{index}")
    try:
        return original_backward(*args, **kwargs)
    finally:
        if phase == "timeline":
            torch.cuda.nvtx.range_pop()
        mark(f"backward:{index}:end")
        active["backwards"] += 1

def prefetch(handle, *args, **kwargs):
    global stage
    previous = stage
    _, target = previous
    assert previous[0] == "forward" and target == handle.metadata["microbatch"] + 1
    assert handle.metadata["step"] == active["iteration"]
    rng = recompute._get_all_rng_states() if phase != "perf" else None
    stage = ("prefetch", handle.metadata["microbatch"])
    key = f"replay:{handle.checkpoint_id}:{target}"
    mark(key + ":start")
    try:
        result = original_prefetch(handle, *args, **kwargs)
        if rng is not None:
            cpu, cuda, tracker = recompute._get_all_rng_states()
            assert torch.equal(cpu, rng[0]) and torch.equal(cuda, rng[1])
            assert tracker.keys() == rng[2].keys()
            assert all(torch.equal(value, rng[2][key]) for key, value in tracker.items())
        if result and phase == "timeline":
            for name in ("prefetch_start", "prefetch_complete"):
                events.append((f"{key}:{name}", None, handle._cuda_events[name]))
        active["max_cached"] = max(active["max_cached"], len(handle._cached_ids))
        return result
    finally:
        mark(key + ":end")
        stage = previous

def cleanup(handle):
    original_cleanup(handle)
    active["released"] += 1
    hit = any(e["event"] == "cache_hit" for e in handle.events)
    active["hits"] += int(hit)
    active["checkpoints"].append(dict(id=handle.checkpoint_id, hit=hit,
        metadata=handle.metadata, events=handle.events, final_state=handle.state))
    if hit and phase == "timeline":
        event = handle._cuda_events["backward_consume"]
        events.append((f"consume:{handle.checkpoint_id}", None, event))
        mark(f"release:{handle.checkpoint_id}")

bucket_type = param_and_grad_buffer._ParamAndGradBucketGroup
original_sync = bucket_type.start_grad_sync

def sync(group, *args, **kwargs):
    if active is not None:
        active["sync"].append(dict(stage=stage, last=group.is_last_microbatch,
                                   async_enabled=group.ddp_config.overlap_grad_reduce))
        assert group.is_last_microbatch, (rank, stage)
    return original_sync(group, *args, **kwargs)

original_rs = param_and_grad_buffer.dist_reduce_scatter_func

def reduce_scatter(*args, **kwargs):
    if active is not None:
        active["reduce_scatter"].append(dict(stage=stage, async_op=kwargs["async_op"]))
    return original_rs(*args, **kwargs)

def train_step(*args, **kwargs):
    global active, stage
    calls.clear()
    events.clear()
    stage = None
    active = dict(iteration=kwargs["iteration"], backwards=0, released=0, hits=0,
                  checkpoints=[], sync=[], max_cached=0, reduce_scatter=[])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        result = original_train_step(*args, **kwargs)
        active.update(cpu_step_ms=(time.perf_counter()-start)*1000,
                      allocated=torch.cuda.max_memory_allocated(),
                      reserved=torch.cuda.max_memory_reserved(),
                      qkv_calls=sum(calls.values()), calls=dict(calls),
                      losses={name: value.item() for name, value in result[0].items()})
        if events:
            torch.cuda.synchronize()
            zero = events[0][2]
            active["timeline"] = [(name, cpu, zero.elapsed_time(event))
                                  for name, cpu, event in events]
        steps.append(active)
        return result
    finally:
        active = None

TELayerNormColumnParallelLinear.forward = qkv
attention.SelfAttention.get_query_key_value_tensors = get_qkv
schedules.forward_step = forward
schedules.backward_step = backward
recompute.ProactiveCheckpoint.prefetch = prefetch
recompute.ProactiveCheckpoint._cleanup_after_backward = cleanup
bucket_type.start_grad_sync = sync
param_and_grad_buffer.dist_reduce_scatter_func = reduce_scatter
training.train_step = train_step
# Keep detailed records in the sidecar and avoid synchronous console logging in timing runs.
if phase == "perf":
    logging.getLogger(recompute.__name__).setLevel(logging.WARNING)
delegate = (out.parents[2].parent / "qwen4b_c" / "gradient_entry.py"
            if phase in ("single", "tp2", "dp4") else repo / "pretrain_gpt.py")
try:
    sys.argv[0] = str(delegate)
    runpy.run_path(str(delegate), run_name="__main__")
finally:
    (out / f"rank{rank}.json").write_text(json.dumps(dict(steps=steps), indent=2))
