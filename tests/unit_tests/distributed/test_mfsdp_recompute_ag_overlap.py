# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import PrefetchOrder


def _bare_fsdp_for_prefetch(use_hybrid_fsdp: bool, outer_strategy: str) -> MegatronFSDP:
    fsdp = object.__new__(MegatronFSDP)
    object.__setattr__(fsdp, "all_gather_pipeline", Mock())
    object.__setattr__(fsdp, "suggested_AG_prefetch_size", 128)
    object.__setattr__(fsdp, "dist_index", SimpleNamespace(use_hybrid_fsdp=use_hybrid_fsdp))
    object.__setattr__(
        fsdp, "ddp_config", SimpleNamespace(outer_dp_sharding_strategy=outer_strategy)
    )
    object.__setattr__(fsdp, "microbatch_count", 0)
    object.__setattr__(fsdp, "model_auto_sync", False)
    return fsdp


def test_mfsdp_recompute_ag_overlap_is_disabled_by_default():
    """The recompute-specific prefetch must not affect existing FSDP users by default."""
    assert not DistributedDataParallelConfig().megatron_fsdp_recompute_ag_overlap


def test_mfsdp_recompute_prefetch_uses_backward_order():
    """Recompute prefetch targets the next FSDP unit consumed by backward."""
    fsdp = _bare_fsdp_for_prefetch(use_hybrid_fsdp=True, outer_strategy="optim")
    params = [Mock()]

    fsdp._prefetch_params_for_recompute(params)

    fsdp.all_gather_pipeline.all_gather_params.assert_called_once_with(
        params=params,
        prefetch=True,
        prefetch_order=PrefetchOrder.BACKWARD_PASS_ORDER,
        suggested_AG_prefetch_size=1,
        outer_fsdp_group_param_gather=True,
    )


def test_recompute_grad_sync_overlap_enables_async_gradient_reduce():
    """The explicit recompute mode reuses the proven asynchronous bucket protocol."""
    config = DistributedDataParallelConfig(recompute_grad_sync_overlap=True)
    assert config.overlap_grad_reduce
    assert config.recompute_grad_sync_overlap
