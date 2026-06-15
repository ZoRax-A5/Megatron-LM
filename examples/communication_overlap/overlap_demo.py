#!/usr/bin/env python3
"""Minimal CUDA compute and NCCL communication overlap experiment."""

import argparse
import os
import random
import statistics
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare serial execution with compute/communication overlap on multiple GPUs."
        )
    )
    parser.add_argument(
        "--collective",
        choices=("all_reduce", "all_gather"),
        default="all_reduce",
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--compute-repeats", type=int, default=8)
    parser.add_argument("--comm-size-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of randomized measurement rounds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used to randomize mode order identically on every rank.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate correctness of all modes and exit without timing measurements.",
    )
    return parser.parse_args()


def initialize_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if dist.get_world_size() < 2:
        raise RuntimeError("Run with at least two processes/GPUs.")

    return local_rank


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        self.default_stream = torch.cuda.current_stream()
        self.comm_stream = torch.cuda.Stream()

        matrix_shape = (args.matrix_size, args.matrix_size)
        self.left = torch.randn(matrix_shape, device=self.device, dtype=self.dtype)
        self.right = torch.randn(matrix_shape, device=self.device, dtype=self.dtype)
        self.compute_output = torch.empty(matrix_shape, device=self.device, dtype=self.dtype)

        element_size = torch.empty((), dtype=self.dtype).element_size()
        comm_elements = args.comm_size_mb * 1024 * 1024 // element_size
        self.comm_input = torch.empty(comm_elements, device=self.device, dtype=self.dtype)
        if args.collective == "all_gather":
            self.comm_output = torch.empty(
                comm_elements * self.world_size, device=self.device, dtype=self.dtype
            )
        else:
            self.comm_output = None

        self._reset_communication_input()
        torch.cuda.synchronize()

    def _reset_communication_input(self):
        self.comm_input.fill_(self.rank + 1)

    def _launch_collective(self, async_op):
        if self.args.collective == "all_reduce":
            return dist.all_reduce(self.comm_input, async_op=async_op)

        return dist.all_gather_into_tensor(
            self.comm_output,
            self.comm_input,
            async_op=async_op,
        )

    def _compute(self):
        for _ in range(self.args.compute_repeats):
            torch.mm(self.left, self.right, out=self.compute_output)

    def _prepare_iteration(self):
        self._reset_communication_input()
        torch.cuda.synchronize()
        dist.barrier()

    def run(self, mode):
        self._prepare_iteration()
        start = time.perf_counter()
        work = None

        with nvtx_range(mode):
            if mode == "compute_only":
                with nvtx_range("compute"):
                    self._compute()

            elif mode == "comm_only":
                with nvtx_range("communication"):
                    work = self._launch_collective(async_op=True)
                    work.wait()

            elif mode == "serial":
                # The synchronous API adds the dependency before compute is enqueued.
                with nvtx_range("communication"):
                    work = self._launch_collective(async_op=False)
                with nvtx_range("compute"):
                    self._compute()

            elif mode == "premature_wait":
                # The collective is asynchronous, but waiting immediately inserts a
                # dependency before compute and removes the overlap opportunity.
                with nvtx_range("communication_then_immediate_wait"):
                    work = self._launch_collective(async_op=True)
                    work.wait()
                with nvtx_range("compute_after_wait"):
                    self._compute()

            elif mode == "overlap":
                # ProcessGroupNCCL executes the collective on an internal NCCL stream.
                # Enqueue independent compute before waiting for the collective result.
                with nvtx_range("communication_async"):
                    work = self._launch_collective(async_op=True)
                with nvtx_range("compute_before_wait"):
                    self._compute()
                work.wait()

            elif mode == "explicit_stream_overlap":
                # This version makes dependencies explicit with a user communication
                # stream and an event. ProcessGroupNCCL still owns its internal stream;
                # work.wait() links its completion to the active user stream.
                comm_done = torch.cuda.Event()
                with torch.cuda.stream(self.comm_stream):
                    with nvtx_range("communication_on_comm_stream"):
                        work = self._launch_collective(async_op=True)
                        work.wait()
                    comm_done.record()

                with nvtx_range("compute_on_default_stream"):
                    self._compute()

                self.default_stream.wait_event(comm_done)

            else:
                raise ValueError(f"Unknown mode: {mode}")

        torch.cuda.synchronize()
        local_seconds = time.perf_counter() - start

        # Report the slowest rank because a distributed step completes at that speed.
        elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=self.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item() * 1000.0

    def validate(self):
        """Validate correctness using both the default-stream and explicit-stream overlap patterns."""
        # --- default-stream overlap pattern ---
        self._prepare_iteration()
        work = self._launch_collective(async_op=True)
        self._compute()
        work.wait()
        torch.cuda.synchronize()

        self._check_results("overlap")

        # --- explicit-stream overlap pattern ---
        self._prepare_iteration()
        comm_done = torch.cuda.Event()
        with torch.cuda.stream(self.comm_stream):
            work = self._launch_collective(async_op=True)
            work.wait()
            comm_done.record()
        self._compute()
        self.default_stream.wait_event(comm_done)
        torch.cuda.synchronize()

        self._check_results("explicit_stream_overlap")

    def _check_results(self, pattern_name):
        if self.args.collective == "all_reduce":
            expected = self.world_size * (self.world_size + 1) / 2
            actual = self.comm_input[0].float().item()
            if actual != expected:
                raise AssertionError(
                    f"[{pattern_name}] all_reduce result {actual} != expected {expected}"
                )
        else:
            chunks = self.comm_output.view(self.world_size, -1)
            actual = chunks[:, 0].float()
            expected = torch.arange(
                1, self.world_size + 1, device=self.device, dtype=torch.float32
            )
            torch.testing.assert_close(actual, expected)

        if not torch.isfinite(self.compute_output).all():
            raise AssertionError(
                f"[{pattern_name}] Compute output contains non-finite values."
            )


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values):
    median = statistics.median(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "median": median,
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "cv": stdev / mean if mean else 0.0,
    }


def measure_randomized_rounds(experiment, modes, warmup, iterations, seed):
    measurements = {mode: [] for mode in modes}

    # Every rank constructs the same order without an extra synchronization.
    # Interleaving modes prevents slow clock/temperature drift from favoring
    # whichever mode happens to run first or last.
    for round_index in range(warmup + iterations):
        order = list(modes)
        random.Random(seed + round_index).shuffle(order)
        is_warmup = round_index < warmup
        for mode in order:
            elapsed = experiment.run(mode)
            if not is_warmup:
                measurements[mode].append(elapsed)

    return measurements


def print_results(args, local_rank, measurements):
    summaries = {mode: summarize(values) for mode, values in measurements.items()}
    serial = measurements["serial"]
    premature = measurements["premature_wait"]
    overlap = measurements["overlap"]
    comm = measurements["comm_only"]

    # Primary baseline: premature_wait uses the same async_op=True API as
    # overlap but with an immediate wait, producing a GPU-serialized schedule
    # without CPU-side blocking.  serial (async_op=False) blocks the CPU until
    # NCCL finishes, which may add a small CPU wake-up gap; it is reported
    # separately for completeness.
    paired_speedups = [
        premature_ms / overlap_ms
        for premature_ms, overlap_ms in zip(premature, overlap)
    ]
    paired_speedups_vs_serial = [
        serial_ms / overlap_ms for serial_ms, overlap_ms in zip(serial, overlap)
    ]
    hidden_fractions = [
        max(0.0, min(1.0, (premature_ms - overlap_ms) / comm_ms))
        for premature_ms, overlap_ms, comm_ms in zip(premature, overlap, comm)
    ]

    print()
    print(f"GPUs:             {dist.get_world_size()}")
    print(f"GPU 0:            {torch.cuda.get_device_name(local_rank)}")
    print(f"Collective:       {args.collective}")
    print(f"Rounds:           {args.iterations} randomized")
    print(f"CUDA connections: {os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'unset')}")
    print()
    print("Mode                       median ms       [p10, p90] ms       CV")
    print("-" * 69)
    labels = {
        "compute_only": "Compute only",
        "comm_only": "Communication only",
        "serial": "Serial (sync API)",
        "premature_wait": "Async, early wait",
        "overlap": "Delayed-wait overlap",
        "explicit_stream_overlap": "Explicit streams",
    }
    for mode, label in labels.items():
        result = summaries[mode]
        print(
            f"{label:<27}"
            f"{result['median']:>9.3f}       "
            f"[{result['p10']:>6.3f}, {result['p90']:>6.3f}]"
            f"     {result['cv'] * 100:>5.1f}%"
        )

    compute_median = summaries["compute_only"]["median"]
    comm_median = summaries["comm_only"]["median"]
    ideal_bound = max(compute_median, comm_median)
    workload_balance = min(compute_median, comm_median) / max(compute_median, comm_median)
    speedup_summary = summarize(paired_speedups)
    speedup_serial_summary = summarize(paired_speedups_vs_serial)
    hidden_summary = summarize(hidden_fractions)

    print()
    print(f"Ideal overlap bound from medians: {ideal_bound:.3f} ms")
    print(f"Workload balance (smaller/larger): {workload_balance:.3f}")
    print(
        "Paired early-wait/overlap speedup: "
        f"{speedup_summary['median']:.3f}x "
        f"[p10 {speedup_summary['p10']:.3f}x, p90 {speedup_summary['p90']:.3f}x]"
    )
    print(
        "  (vs sync serial:                "
        f"{speedup_serial_summary['median']:.3f}x "
        f"[p10 {speedup_serial_summary['p10']:.3f}x, p90 {speedup_serial_summary['p90']:.3f}x])"
    )
    print(
        "Paired communication hidden:      "
        f"{hidden_summary['median'] * 100:.1f}% "
        f"[p10 {hidden_summary['p10'] * 100:.1f}%, "
        f"p90 {hidden_summary['p90'] * 100:.1f}%]"
    )
    print()
    if workload_balance < 0.5:
        print("WARNING: Compute and communication are poorly balanced. Tune")
        print("--compute-repeats or --comm-size-mb so their standalone medians")
        print("are within roughly 2x; otherwise the overlap signal is easy to")
        print("lose in normal timing noise.")
        print()
    print("The primary baseline is early-wait/overlap — both use async_op=True, so the")
    print("comparison isolates the effect of scheduling the wait after compute.  The")
    print("sync-serial comparison is shown for reference but may include CPU wake-up gaps.")
    print()
    print("Low CV and a paired speedup consistently above 1.0 are stronger evidence")
    print("than a single timing. Use Nsight Systems to prove kernel overlap.")


def main():
    args = parse_args()
    local_rank = initialize_distributed()
    experiment = Experiment(args)

    if args.validate_only:
        experiment.validate()
        if dist.get_rank() == 0:
            print("All validation checks passed.")
        dist.barrier()
        dist.destroy_process_group()
        return

    modes = (
        "compute_only",
        "comm_only",
        "serial",
        "premature_wait",
        "overlap",
        "explicit_stream_overlap",
    )
    measurements = measure_randomized_rounds(
        experiment,
        modes,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
    )
    experiment.validate()

    if dist.get_rank() == 0:
        print_results(args, local_rank, measurements)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
