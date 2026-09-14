# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
import torch

from megatron.core.tensor_parallel.layers import linear_with_frozen_weight
from megatron.core.tensor_parallel.mappings import (
    launch_all_gather_from_sequence_parallel_region,
    launch_reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.recompute_comm_overlap import (
    TensorParallelRecomputeCommOverlapConfig,
    assert_tensor_parallel_recompute_is_collective_free,
    finish_tensor_parallel_collective_with_active_recompute,
    overlap_with_pending_tensor_parallel_collective,
    tensor_parallel_recompute_comm_overlap,
)
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils


class _Pending:
    def __init__(self, trace, *, collective="all_gather", supports_compute_overlap=True):
        self.trace = trace
        self.collective = collective
        self.supports_compute_overlap = supports_compute_overlap

    def wait(self):
        self.trace.append("wait")
        return "collective-output"


def _enabled_config(**kwargs):
    values = {
        "recompute_comm_overlap": True,
        "tensor_parallel_overlap": True,
        "sequence_parallel": True,
    }
    values.update(kwargs)
    return TensorParallelRecomputeCommOverlapConfig(**values)


def test_overlap_orders_recompute_between_launch_and_wait():
    trace = ["launch"]
    pending = _Pending(trace)

    output, recomputed = overlap_with_pending_tensor_parallel_collective(
        pending, lambda: trace.append("recompute") or "recomputed-output"
    )

    assert trace == ["launch", "recompute", "wait"]
    assert output == "collective-output"
    assert recomputed == "recomputed-output"


def test_non_nccl_collective_is_waited_before_recompute():
    trace = ["launch"]
    pending = _Pending(trace, supports_compute_overlap=False)

    overlap_with_pending_tensor_parallel_collective(
        pending, lambda: trace.append("recompute")
    )

    assert trace == ["launch", "wait", "recompute"]


def test_recompute_error_still_drains_pending_collective():
    trace = ["launch"]
    pending = _Pending(trace)

    def fail_recompute():
        trace.append("recompute")
        raise RuntimeError("recompute failed")

    with pytest.raises(RuntimeError, match="recompute failed"):
        overlap_with_pending_tensor_parallel_collective(pending, fail_recompute)

    assert trace == ["launch", "recompute", "wait"]


def test_context_consumes_callback_at_most_once():
    trace = []
    with tensor_parallel_recompute_comm_overlap(
        _enabled_config(), lambda: trace.append("recompute")
    ):
        finish_tensor_parallel_collective_with_active_recompute(_Pending(trace))
        finish_tensor_parallel_collective_with_active_recompute(
            _Pending(trace, collective="reduce_scatter")
        )

    assert trace == ["recompute", "wait", "wait"]


def test_context_rejects_tp_collective_in_recompute_callback():
    trace = []

    with tensor_parallel_recompute_comm_overlap(
        _enabled_config(), assert_tensor_parallel_recompute_is_collective_free
    ):
        with pytest.raises(RuntimeError, match="must be TP-collective-free"):
            finish_tensor_parallel_collective_with_active_recompute(_Pending(trace))

    assert trace == ["wait"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"recompute_comm_overlap": False},
            "requires the recompute_comm_overlap master switch",
        ),
        ({"sequence_parallel": False}, "requires sequence parallelism"),
        ({"use_transformer_engine": True}, "does not yet support Transformer Engine"),
        (
            {"overlap_all_gather": False, "overlap_reduce_scatter": False},
            "must enable All-Gather or Reduce-Scatter",
        ),
    ],
)
def test_config_rejects_unsupported_domains(overrides, message):
    with pytest.raises(ValueError, match=message):
        _enabled_config(**overrides).validate()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_phase_collectives_are_autograd_safe_on_two_tp_ranks():
    if Utils.world_size < 2:
        pytest.skip("requires at least two distributed ranks")

    Utils.initialize_model_parallel(2, 1)
    try:
        tp_group = get_tensor_model_parallel_group_if_none(None)
        tp_rank = tp_group.rank()
        tp_size = tp_group.size()
        config = _enabled_config()
        recomputed = []

        gather_input = torch.full(
            (2, 3), float(tp_rank + 1), device="cuda", requires_grad=True
        )
        gather_pending = launch_all_gather_from_sequence_parallel_region(
            gather_input, group=tp_group
        )
        gather_output, _ = overlap_with_pending_tensor_parallel_collective(
            gather_pending,
            lambda: recomputed.append(torch.arange(128, device="cuda").square().sum()),
        )
        expected_gather = torch.cat(
            [
                torch.full((2, 3), float(rank + 1), device="cuda")
                for rank in range(tp_size)
            ]
        )
        assert torch.equal(gather_output, expected_gather)
        gather_output.sum().backward()
        assert torch.equal(gather_input.grad, torch.full_like(gather_input, float(tp_size)))

        reduce_scatter_input = (
            torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3) + tp_rank
        ).requires_grad_(True)
        reduce_scatter_pending = launch_reduce_scatter_to_sequence_parallel_region(
            reduce_scatter_input, group=tp_group
        )
        reduce_scatter_output, _ = overlap_with_pending_tensor_parallel_collective(
            reduce_scatter_pending,
            lambda: recomputed.append(torch.arange(128, device="cuda").sin().sum()),
        )
        rank_offset = sum(range(tp_size))
        expected_reduce_scatter = (
            torch.arange(12, device="cuda", dtype=torch.float32)
            .reshape(4, 3)[tp_rank * 2 : (tp_rank + 1) * 2]
            .mul(tp_size)
            .add(rank_offset)
        )
        assert torch.equal(reduce_scatter_output, expected_reduce_scatter)
        reduce_scatter_output.sum().backward()
        assert torch.equal(reduce_scatter_input.grad, torch.ones_like(reduce_scatter_input))
        assert len(recomputed) == 2

        # Exercise the native linear call-site, not just the mapping primitive.
        linear_input = torch.ones((2, 4), device="cuda", requires_grad=True)
        weight = torch.ones((3, 4), device="cuda")
        callbacks = []
        with tensor_parallel_recompute_comm_overlap(
            config, lambda: callbacks.append(torch.ones(128, device="cuda").square().sum())
        ):
            linear_output = linear_with_frozen_weight(
                linear_input,
                weight,
                bias=None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=True,
                tp_group=tp_group,
            )
        linear_output.sum().backward()
        assert torch.equal(
            linear_input.grad,
            torch.full_like(linear_input, float(weight.shape[0] * tp_size)),
        )
        assert len(callbacks) == 1
    finally:
        Utils.destroy_model_parallel()
