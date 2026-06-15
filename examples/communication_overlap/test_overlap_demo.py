#!/usr/bin/env python3
"""Automated correctness tests for the communication-overlap demo.

Run with two or more GPUs::

    torchrun --standalone --nproc-per-node=2 \\
        examples/communication_overlap/test_overlap_demo.py

The tests validate numerical correctness of all modes (including the overlap
patterns) for both all_reduce and all_gather collectives.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# Allow importing overlap_demo from the same directory.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import overlap_demo  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run correctness tests for the overlap demo."
    )
    parser.add_argument(
        "--collective",
        choices=("all_reduce", "all_gather"),
        default="all_reduce",
    )
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--compute-repeats", type=int, default=4)
    parser.add_argument("--comm-size-mb", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


class OverlapDemoTest:
    """Thin wrapper that creates an Experiment and runs assertions."""

    def __init__(self):
        self.args = parse_args()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for overlap demo tests.")

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if dist.get_world_size() < 2:
            raise RuntimeError("Run with at least two processes/GPUs.")

        # Create a namespace-like object that overlap_demo.Experiment expects.
        ns = argparse.Namespace(
            collective=self.args.collective,
            matrix_size=self.args.matrix_size,
            compute_repeats=self.args.compute_repeats,
            comm_size_mb=self.args.comm_size_mb,
            dtype=self.args.dtype,
        )
        self.experiment = overlap_demo.Experiment(ns)
        self.rank = dist.get_rank()

    def test_validate_overlap(self):
        """validate() already covers both default-stream and explicit-stream patterns."""
        try:
            self.experiment.validate()
        except AssertionError as e:
            raise AssertionError(f"validate() failed: {e}") from e

    def test_all_modes_run(self):
        """Every mode must return a finite, positive elapsed time."""
        modes = (
            "compute_only",
            "comm_only",
            "serial",
            "premature_wait",
            "overlap",
            "explicit_stream_overlap",
        )
        for mode in modes:
            elapsed = self.experiment.run(mode)
            if elapsed <= 0:
                raise AssertionError(f"Mode '{mode}' returned non-positive time: {elapsed}")
            if not (elapsed > 0 and elapsed < float("inf")):
                raise AssertionError(f"Mode '{mode}' returned non-finite time: {elapsed}")

    def test_overlap_not_catastrophically_worse(self):
        """Overlap should not be dramatically slower than the no-overlap baseline."""
        modes = ("premature_wait", "overlap")
        measurements = {m: [] for m in modes}
        for _ in range(5):
            for m in modes:
                measurements[m].append(self.experiment.run(m))

        premature_median = sorted(measurements["premature_wait"])[len(measurements["premature_wait"]) // 2]
        overlap_median = sorted(measurements["overlap"])[len(measurements["overlap"]) // 2]

        # Overlap should not be more than 2× worse than the no-overlap baseline.
        # (It may not be faster on all hardware, but it shouldn't be much slower.)
        ratio = overlap_median / premature_median
        if ratio > 2.0:
            raise AssertionError(
                f"Overlap median ({overlap_median:.3f} ms) is more than 2× worse "
                f"than premature_wait median ({premature_median:.3f} ms). "
                f"Ratio: {ratio:.3f}"
            )

    def run_all(self):
        self.test_validate_overlap()
        if self.rank == 0:
            print("PASS: test_validate_overlap")

        self.test_all_modes_run()
        if self.rank == 0:
            print("PASS: test_all_modes_run")

        self.test_overlap_not_catastrophically_worse()
        if self.rank == 0:
            print("PASS: test_overlap_not_catastrophically_worse")


def main():
    test = OverlapDemoTest()
    test.run_all()
    if dist.get_rank() == 0:
        print()
        print("All overlap demo tests passed.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
