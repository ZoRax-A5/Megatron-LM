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

from contextlib import contextmanager

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
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
import megatron.training.training as training

steps = []
calls = Counter()
events = []
stage = None
current = None
active = None
original_train_step = training.train_step
original_qkv = TELayerNormColumnParallelLinear.forward
profile_step = int(os.environ.get("P10_CAPTURE_STEP", "2"))
poc_mode = os.environ.get("P10_POC", "")

@contextmanager
def trace_range(kind, **metadata):
    enabled = phase == "timeline" and active is not None and active["iteration"] == profile_step
    if enabled:
        label = dict(kind=kind, rank=rank, step=profile_step, stage=stage, **metadata)
        torch.cuda.nvtx.range_push("P10|" + json.dumps(label, separators=(",", ":")))
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()

original_forward = schedules.forward_step
original_backward = schedules.backward_step
original_prefetch = recompute.ProactiveCheckpoint.prefetch
original_cleanup = recompute.ProactiveCheckpoint._cleanup_after_backward

def qkv(module, *args, **kwargs):
    if hasattr(module, "_p1tag") and module._p1tag[1] != "qkv":
        return original_qkv(module, *args, **kwargs)
    # Tag the fused QKV modules from SelfAttention; FC1 uses the same TE type.
    calls[str(stage)] += int(id(module) in qkv_modules)
    if id(module) not in qkv_modules:
        return original_qkv(module, *args, **kwargs)
    with trace_range("qkv", layer=qkv_modules[id(module)]):
        return original_qkv(module, *args, **kwargs)

qkv_modules = {}
original_get_qkv = attention.SelfAttention.get_query_key_value_tensors

def get_qkv(module, *args, **kwargs):
    qkv_modules[id(module.linear_qkv)] = module.layer_number
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
    active["schedule"].append(stage)
    mark(f"forward:{current}:start")
    if phase == "timeline":
        torch.cuda.nvtx.range_push(f"p9.forward.{current}")
    try:
        with trace_range("forward"):
            return original_forward(*args, **kwargs)
    finally:
        if phase == "timeline":
            torch.cuda.nvtx.range_pop()
        mark(f"forward:{current}:end")

def backward(*args, **kwargs):
    global stage
    index = active["backwards"]
    stage = ("backward", index)
    active["schedule"].append(stage)
    mark(f"backward:{index}:start")
    if phase == "timeline":
        torch.cuda.nvtx.range_push(f"p9.backward.{index}")
    try:
        with trace_range("backward"):
            return original_backward(*args, **kwargs)
    finally:
        if phase == "timeline":
            torch.cuda.nvtx.range_pop()
        mark(f"backward:{index}:end")
        active["backwards"] += 1

def prefetch(handle, *args, **kwargs):
    global stage
    previous = stage
    assert previous == ("backward", 7)
    active["trigger_count"] += 1
    if poc_mode == "Cnoop":
        with trace_range("noop", checkpoint_id=handle.checkpoint_id):
            return False
    _, target = previous
    if previous[0] == "forward":
        assert target == handle.metadata["microbatch"] + 1
    else:
        assert previous == ("backward", 7) and target == handle.metadata["microbatch"]
    assert handle.metadata["step"] == active["iteration"]
    rng = recompute._get_all_rng_states() if phase != "perf" else None
    stage = ("prefetch", handle.metadata["microbatch"])
    key = f"replay:{handle.checkpoint_id}:{target}"
    mark(key + ":start")
    try:
        with trace_range("prefetch", checkpoint_id=handle.checkpoint_id,
                         trigger_mb=target, layer=handle.metadata["layer_number"]):
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

original_ddp_init = DistributedDataParallel.__init__

def ddp_init(ddp, *args, **kwargs):
    original_ddp_init(ddp, *args, **kwargs)
    for ordinal, group in enumerate(ddp.bucket_groups):
        group._p11_group_ordinal = ordinal

DistributedDataParallel.__init__ = ddp_init

bucket_type = param_and_grad_buffer._ParamAndGradBucketGroup
original_sync = bucket_type.start_grad_sync

def sync(group, *args, **kwargs):
    if active is not None:
        active["sync"].append(dict(stage=stage, group=group._p11_group_ordinal,
                                   last=group.is_last_microbatch,
                                   async_enabled=group.ddp_config.overlap_grad_reduce))
        assert group.is_last_microbatch, (rank, stage)
    with trace_range("grad_sync", group=group._p11_group_ordinal):
        result = original_sync(group, *args, **kwargs)
    return result

original_finish = bucket_type.finish_grad_sync

def finish(group, *args, **kwargs):
    try:
        with trace_range("grad_finish", group=group._p11_group_ordinal):
            return original_finish(group, *args, **kwargs)
    except AssertionError:
        print("P10_DDP_COUNTER", rank,
              [(tuple(param.shape), group.golden_per_param_grad_ready_counts.get(param),
                group.per_param_grad_ready_counts.get(param))
               for param in group.params],
              
              flush=True)
        raise

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
                  checkpoints=[], sync=[], max_cached=0, reduce_scatter=[], schedule=[], trigger_count=0)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    capture = phase == "timeline" and active["iteration"] == profile_step
    if capture:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.profiler.start()
    optimizer = args[3]
    original_opt = optimizer.step
    def observe_optimizer(*opt_args, **opt_kwargs):
        with trace_range("optimizer"):
            return original_opt(*opt_args, **opt_kwargs)
    if capture:
        optimizer.step = observe_optimizer
    try:
        with trace_range("step"):
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
        if capture:
            optimizer.step = original_opt
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        active = None

TELayerNormColumnParallelLinear.forward = qkv
attention.SelfAttention.get_query_key_value_tensors = get_qkv
schedules.forward_step = forward
schedules.backward_step = backward
recompute.ProactiveCheckpoint.prefetch = prefetch
recompute.ProactiveCheckpoint._cleanup_after_backward = cleanup
bucket_type.start_grad_sync = sync
bucket_type.finish_grad_sync = finish
param_and_grad_buffer.dist_reduce_scatter_func = reduce_scatter
training.train_step = train_step

original_all_reduce = torch.distributed.all_reduce

def all_reduce(*args, **kwargs):
    if phase != "timeline" or active is None or active["iteration"] != profile_step:
        return original_all_reduce(*args, **kwargs)
    group = kwargs.get("group", args[2] if len(args)>2 else None)
    peers = torch.distributed.get_process_group_ranks(
        group if group is not None else torch.distributed.group.WORLD
    )
    with trace_range("all_reduce_api", peers=peers):
        return original_all_reduce(*args, **kwargs)

torch.distributed.all_reduce = all_reduce
# Keep detailed records in the sidecar and avoid synchronous console logging in timing runs.
if phase == "perf":
    logging.getLogger(recompute.__name__).setLevel(logging.WARNING)
delegate = (out.parents[2].parent / "qwen4b_c" / "gradient_entry.py"
            if phase in ("single", "tp2", "dp4") else repo / "pretrain_gpt.py")
if os.environ.get("AB_CONVERTED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen4b_ab"))
    import load_weights

try:
    delegate = Path(__file__).parent / "exact_entry.py" if os.environ.get("P10_EXACT") else delegate
    sys.argv[0] = str(delegate)
    runpy.run_path(str(delegate), run_name="__main__")
finally:
    (out / f"rank{rank}.json").write_text(json.dumps(dict(steps=steps), indent=2))
