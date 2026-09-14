# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Two-GPU communication/recomputation coexistence benchmark.

Launch with:
    torchrun --standalone --nproc-per-node=2 tools/recompute_comm_overlap_benchmark.py
"""

import argparse
import json
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist

PRIMITIVES = ("p2p", "all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", nargs="+", choices=PRIMITIVES, default=list(PRIMITIVES))
    parser.add_argument("--message-mb", type=int, default=64)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--compute-iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json-output", type=str)
    return parser.parse_args()


def _max_rank_seconds(seconds: float, device: torch.device) -> float:
    value = torch.tensor(seconds, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def _measure(fn: Callable[[], None], device: torch.device) -> float:
    dist.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    return _max_rank_seconds(time.perf_counter() - started, device)


def _wait(work) -> None:
    if isinstance(work, list):
        for item in work:
            item.wait()
    else:
        work.wait()


def _build_collective(
    primitive: str, numel: int, device: torch.device
) -> tuple[Callable[[], object], list[torch.Tensor]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    buffers: list[torch.Tensor] = []

    if primitive == "all_reduce":
        tensor = torch.ones(numel, dtype=dtype, device=device)
        buffers.append(tensor)
        return lambda: dist.all_reduce(tensor, async_op=True), buffers

    if primitive == "all_gather":
        input_tensor = torch.ones(numel, dtype=dtype, device=device)
        output_tensor = torch.empty(numel * world_size, dtype=dtype, device=device)
        buffers.extend((input_tensor, output_tensor))
        return (
            lambda: dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True),
            buffers,
        )

    if primitive == "reduce_scatter":
        input_tensor = torch.ones(numel * world_size, dtype=dtype, device=device)
        output_tensor = torch.empty(numel, dtype=dtype, device=device)
        buffers.extend((input_tensor, output_tensor))
        return (
            lambda: dist.reduce_scatter_tensor(output_tensor, input_tensor, async_op=True),
            buffers,
        )

    if primitive == "all_to_all":
        input_tensor = torch.ones(numel * world_size, dtype=dtype, device=device)
        output_tensor = torch.empty_like(input_tensor)
        buffers.extend((input_tensor, output_tensor))
        return (
            lambda: dist.all_to_all_single(output_tensor, input_tensor, async_op=True),
            buffers,
        )

    if primitive == "p2p":
        send = torch.ones(numel, dtype=dtype, device=device)
        recv = torch.empty_like(send)
        peer = (rank + 1) % world_size
        buffers.extend((send, recv))

        def launch_p2p():
            operations = [
                dist.P2POp(dist.isend, send, peer),
                dist.P2POp(dist.irecv, recv, peer),
            ]
            return dist.batch_isend_irecv(operations)

        return launch_p2p, buffers

    raise ValueError(f"Unsupported primitive: {primitive}")


def _benchmark_primitive(
    primitive: str,
    args: argparse.Namespace,
    device: torch.device,
    comm_stream: torch.cuda.Stream,
) -> dict[str, float | str]:
    dtype_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    numel = args.message_mb * 1024 * 1024 // dtype_bytes
    launch_collective, communication_buffers = _build_collective(primitive, numel, device)
    lhs = torch.randn(args.matrix_size, args.matrix_size, dtype=torch.bfloat16, device=device)
    rhs = torch.randn_like(lhs)
    result = torch.empty_like(lhs)

    def recompute() -> None:
        for _ in range(args.compute_iters):
            torch.mm(lhs, rhs, out=result)

    def communication() -> None:
        with torch.cuda.stream(comm_stream):
            work = launch_collective()
        _wait(work)
        comm_stream.synchronize()

    def sequential() -> None:
        communication()
        recompute()

    def concurrent() -> None:
        with torch.cuda.stream(comm_stream):
            work = launch_collective()
        recompute()
        _wait(work)
        comm_stream.synchronize()

    for _ in range(args.warmup):
        sequential()
        concurrent()
    torch.cuda.synchronize(device)

    samples = {"communication_ms": 0.0, "recompute_ms": 0.0, "sequential_ms": 0.0, "concurrent_ms": 0.0}
    measurements = {
        "communication_ms": communication,
        "recompute_ms": recompute,
        "sequential_ms": sequential,
        "concurrent_ms": concurrent,
    }
    for name, fn in measurements.items():
        elapsed = sum(_measure(fn, device) for _ in range(args.iters)) / args.iters
        samples[name] = elapsed * 1000.0

    hideable_ms = min(samples["communication_ms"], samples["recompute_ms"])
    samples["hidden_fraction"] = (
        max(0.0, samples["sequential_ms"] - samples["concurrent_ms"]) / hideable_ms
        if hideable_ms > 0
        else 0.0
    )
    dominant_ms = max(samples["communication_ms"], samples["recompute_ms"])
    samples["concurrent_interference"] = (
        samples["concurrent_ms"] / dominant_ms - 1.0 if dominant_ms > 0 else 0.0
    )
    samples["primitive"] = primitive

    # Keep allocations alive until all timed kernels have completed.
    del communication_buffers, lhs, rhs, result
    return samples


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    comm_stream = torch.cuda.Stream(device=device)
    results = [
        _benchmark_primitive(primitive, args, device, comm_stream)
        for primitive in args.primitives
    ]

    if dist.get_rank() == 0:
        report = {
            "world_size": dist.get_world_size(),
            "gpu": torch.cuda.get_device_name(device),
            "message_mb": args.message_mb,
            "matrix_size": args.matrix_size,
            "compute_iters": args.compute_iters,
            "results": results,
        }
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as output:
                output.write(rendered)
                output.write("\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
