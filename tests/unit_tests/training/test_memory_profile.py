# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import torch

from megatron.training.memory_profile import TrainingMemoryProfiler


class _EmptyModel:
    def parameters(self):
        return []


class _FakeOptimizer:
    def __init__(self):
        self.param_groups = []
        self.state = {}

    def prepare_grads(self):
        return False

    def get_grad_norm(self):
        return 1.0

    def step_with_ready_grads(self):
        return True


def test_memory_profiler_warmup_window_and_optimizer_events(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 200)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 150)
    monkeypatch.setattr(torch.cuda, "device_memory_used", lambda: 250)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    optimizer = _FakeOptimizer()
    profiler = TrainingMemoryProfiler(
        mode="light",
        warmup_iters=2,
        profile_iters=2,
        output_dir=str(tmp_path),
        start_iteration=10,
        model=[_EmptyModel()],
        optimizer=optimizer,
        rank_metadata={"rank": 3, "local_rank": 3},
    )

    profiler.begin_iteration(10)
    profiler.end_iteration(10)
    profiler.begin_iteration(11)
    profiler.end_iteration(11)
    assert not (tmp_path / "rank3.jsonl").read_text().strip()

    profiler.begin_iteration(12)
    optimizer.prepare_grads()
    profiler.end_iteration(12)
    profiler.begin_iteration(13)
    profiler.end_iteration(13)

    records = [json.loads(line) for line in (tmp_path / "rank3.jsonl").read_text().splitlines()]
    assert profiler.finished
    assert [record["iteration"] for record in records] == [12, 12, 12, 12, 13, 13]
    assert [record["event"] for record in records] == [
        "iteration_start",
        "optimizer_prepare_grads_start",
        "optimizer_prepare_grads_end",
        "iteration_end",
        "iteration_start",
        "iteration_end",
    ]
    assert all(record["rank"] == 3 for record in records)
    assert all(record["allocated_bytes"] == 100 for record in records)
