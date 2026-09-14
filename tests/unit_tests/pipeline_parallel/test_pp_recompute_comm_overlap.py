"""Tests for the PP-specific recompute communication opt-in."""

from types import SimpleNamespace

import pytest

from megatron.core.pipeline_parallel.p2p_communication import (
    is_pp_recompute_comm_overlap_enabled,
)


def test_pp_recompute_overlap_is_opt_in():
    """The PP mode remains disabled for legacy configs and global-only configs."""
    assert not is_pp_recompute_comm_overlap_enabled(SimpleNamespace())
    assert not is_pp_recompute_comm_overlap_enabled(
        SimpleNamespace(recompute_comm_overlap=True)
    )


def test_pp_recompute_overlap_requires_global_opt_in():
    """A domain opt-in must not bypass the global safety gate."""
    with pytest.raises(
        ValueError, match="pp_recompute_comm_overlap requires recompute_comm_overlap"
    ):
        is_pp_recompute_comm_overlap_enabled(
            SimpleNamespace(pp_recompute_comm_overlap=True)
        )


def test_pp_recompute_overlap_accepts_both_opt_ins():
    """Both gates enable the asynchronous PP request protocol."""
    assert is_pp_recompute_comm_overlap_enabled(
        SimpleNamespace(recompute_comm_overlap=True, pp_recompute_comm_overlap=True)
    )
