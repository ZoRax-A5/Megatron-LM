"""Fingerprint complete reduced gradient shards and updated parameters per rank."""
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch
import torch.distributed as dist
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
import megatron.training.training as training

repo = Path(os.environ["QWEN_BENCH_MEGATRON"])
out = Path(os.environ["QWEN_BENCH_CASE"])
rank = int(os.environ["RANK"])
checks = []
original_step = training.train_step

def fingerprint(tensors):
    checksum = hashlib.sha256()
    size = 0
    for name, value in sorted(tensors.items()):
        tensor = value.detach()
        checksum.update(json.dumps([name,list(tensor.shape),str(tensor.dtype)]).encode())
        size += tensor.numel()
        for part in tensor.reshape(-1).split(4*1024*1024):
            checksum.update(memoryview(part.cpu().contiguous().view(torch.uint8).numpy()))
    return dict(elements=size, sha256=checksum.hexdigest())

def train_step(*args, **kwargs):
    model, optimizer = args[2:4]
    original_opt = optimizer.step
    row = {"iteration": kwargs["iteration"]}

    def step(*opt_args, **opt_kwargs):
        grads = {}
        for m, ddp in enumerate(model):
            assert isinstance(ddp, DistributedDataParallel)
            for g, group in enumerate(ddp.bucket_groups):
                pg = group.intra_distributed_optimizer_instance_group
                for b, bucket in enumerate(group.buckets):
                    shard = bucket.grad_data.view(-1).chunk(pg.size())[pg.rank()]
                    grads[f"{m}:{g}:{b}"] = shard
        row["reduced_gradients"] = fingerprint(grads)
        result = original_opt(*opt_args, **opt_kwargs)
        weights = {f"{i}:{name}": p for i,m in enumerate(model) for name,p in m.named_parameters()}
        row["parameters"] = fingerprint(weights)
        return result
    optimizer.step = step
    try:
        result = original_step(*args, **kwargs)
        row["loss"] = {key:value.item() for key,value in result[0].items()}
        checks.append(row)
        return result
    finally:
        optimizer.step = original_opt

training.train_step = train_step
try:
    runpy.run_path(str(repo/"pretrain_gpt.py"), run_name="__main__")
finally:
    (out/f"exact_rank{rank}.json").write_text(json.dumps(checks,indent=2))
