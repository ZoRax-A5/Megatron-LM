# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in tensor-parallel communication windows for activation recomputation.

This module deliberately only coordinates an already-launched tensor-parallel
collective and an independent recomputation callback. It does not select which
activation to recompute. The caller owns that dependency decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, TypeVar


class _PendingTensorParallelCollective(Protocol):
    """The part of a pending collective needed by the overlap coordinator."""

    collective: str
    supports_compute_overlap: bool

    def wait(self) -> Any:
        """Wait for the collective and return its autograd-connected output."""


@dataclass(frozen=True)
class TensorParallelRecomputeCommOverlapConfig:
    """Configuration for a native tensor-parallel recompute overlap window.

    Both ``recompute_comm_overlap`` and ``tensor_parallel_overlap`` must be true.
    Keeping the TP switch separate makes this experimental path opt-in even when
    another communication domain uses the global recompute-overlap switch.

    Transformer Engine Userbuffers own their internal launch/wait points, so they
    cannot use this coordinator yet. Only sequence-parallel All-Gather and
    Reduce-Scatter in the native MCore tensor-parallel path are supported.
    """

    recompute_comm_overlap: bool = False
    tensor_parallel_overlap: bool = False
    sequence_parallel: bool = False
    use_transformer_engine: bool = False
    overlap_all_gather: bool = True
    overlap_reduce_scatter: bool = True

    @property
    def enabled(self) -> bool:
        """Whether the master and TP-specific switches both enable the feature."""

        return self.recompute_comm_overlap and self.tensor_parallel_overlap

    def validate(self) -> None:
        """Validate the currently supported overlap domain."""

        if self.tensor_parallel_overlap and not self.recompute_comm_overlap:
            raise ValueError(
                "tensor_parallel_overlap requires the recompute_comm_overlap master switch"
            )
        if not self.enabled:
            return
        if not self.sequence_parallel:
            raise ValueError(
                "tensor-parallel recompute overlap currently requires sequence parallelism"
            )
        if self.use_transformer_engine:
            raise ValueError(
                "tensor-parallel recompute overlap does not yet support Transformer Engine "
                "Userbuffers because their collective launch/wait points are internal"
            )
        if not self.overlap_all_gather and not self.overlap_reduce_scatter:
            raise ValueError(
                "tensor-parallel recompute overlap must enable All-Gather or Reduce-Scatter"
            )

    def allows(self, collective: str) -> bool:
        """Return whether ``collective`` is selected for overlap."""

        if collective == "all_gather":
            return self.overlap_all_gather
        if collective == "reduce_scatter":
            return self.overlap_reduce_scatter
        return False


@dataclass
class _RecomputeOverlapContext:
    config: TensorParallelRecomputeCommOverlapConfig
    recompute: Callable[[], Any]
    callback_consumed: bool = False
    inside_recompute: bool = False


_ACTIVE_RECOMPUTE_OVERLAP: ContextVar[_RecomputeOverlapContext | None] = ContextVar(
    "tensor_parallel_recompute_comm_overlap", default=None
)


@contextmanager
def tensor_parallel_recompute_comm_overlap(
    config: TensorParallelRecomputeCommOverlapConfig, recompute: Callable[[], Any]
) -> Iterator[None]:
    """Make one independent recomputation callback available to a native TP collective.

    The callback must not depend on or mutate the collective input/output, and must
    not launch a tensor-parallel collective itself. A selective LayerNorm or
    elementwise activation recomputation from another microbatch satisfies this
    contract; replaying a full TP transformer block does not.

    Disabled configurations do not install a context and therefore do not consume
    ``recompute``. This preserves the legacy execution path exactly by default.
    """

    config.validate()
    if not config.enabled:
        yield
        return

    if _ACTIVE_RECOMPUTE_OVERLAP.get() is not None:
        raise RuntimeError("tensor-parallel recompute overlap contexts cannot be nested")

    state = _RecomputeOverlapContext(config=config, recompute=recompute)
    token = _ACTIVE_RECOMPUTE_OVERLAP.set(state)
    try:
        yield
    finally:
        _ACTIVE_RECOMPUTE_OVERLAP.reset(token)


def is_tensor_parallel_recompute_overlap_active(collective: str) -> bool:
    """Return whether an active context selected ``collective``."""

    state = _ACTIVE_RECOMPUTE_OVERLAP.get()
    return (
        state is not None
        and not state.callback_consumed
        and state.config.allows(collective)
    )


def assert_tensor_parallel_recompute_is_collective_free() -> None:
    """Reject MCore TP collectives recursively launched by a recompute callback."""

    state = _ACTIVE_RECOMPUTE_OVERLAP.get()
    if state is not None and state.inside_recompute:
        raise RuntimeError(
            "the tensor-parallel overlap recompute callback must be TP-collective-free"
        )


_T = TypeVar("_T")


def overlap_with_pending_tensor_parallel_collective(
    pending: _PendingTensorParallelCollective, recompute: Callable[[], _T]
) -> tuple[Any, _T]:
    """Run independent recomputation between collective launch and wait.

    Non-CUDA/NCCL handles are drained before recomputation. If recomputation
    raises, the pending collective is still drained before the original exception
    is re-raised so that its input and output buffers remain alive long enough.
    """

    if not pending.supports_compute_overlap:
        collective_output = pending.wait()
        return collective_output, recompute()

    try:
        recompute_output = recompute()
    except BaseException as error:
        try:
            pending.wait()
        except BaseException as wait_error:
            error.add_note(f"waiting for the pending TP collective also failed: {wait_error!r}")
        raise

    collective_output = pending.wait()
    return collective_output, recompute_output


def finish_tensor_parallel_collective_with_active_recompute(
    pending: _PendingTensorParallelCollective,
) -> Any:
    """Finish ``pending``, consuming the active callback when it is applicable."""

    state = _ACTIVE_RECOMPUTE_OVERLAP.get()
    if (
        state is None
        or state.callback_consumed
        or not state.config.allows(pending.collective)
        or not pending.supports_compute_overlap
    ):
        return pending.wait()

    state.callback_consumed = True

    def recompute() -> Any:
        state.inside_recompute = True
        try:
            return state.recompute()
        finally:
            state.inside_recompute = False

    output, _ = overlap_with_pending_tensor_parallel_collective(pending, recompute)
    return output
