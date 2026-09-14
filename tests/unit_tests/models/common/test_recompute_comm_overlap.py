# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.transformer import transformer_config as transformer_config_module
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(autouse=True)
def allow_ep_overlap_config_on_older_test_torch(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "is_torch_min_version", lambda _version: True)


class _Node:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def forward(self, value):
        self.trace.append(f"{self.name}.forward")
        return value

    def backward(self, value):
        self.trace.append(f"{self.name}.backward")
        return value

    def backward_dw(self):
        self.trace.append(f"{self.name}.backward_dw")


class _LayerPlan:
    def __init__(self, prefix, trace, recompute_comm_overlap=False):
        self.config = SimpleNamespace(
            recompute_comm_overlap=recompute_comm_overlap,
            ep_overlap_early_attn_memory_release=False,
        )
        self.attn = _Node(f"{prefix}.attn", trace)
        self.moe_dispatch = _Node(f"{prefix}.dispatch", trace)
        self.mlp = _Node(f"{prefix}.mlp", trace)
        self.moe_combine = _Node(f"{prefix}.combine", trace)
        self.mtp_post_process = _Node(f"{prefix}.mtp", trace)

    def get_fp8_context(self):
        return nullcontext()


@pytest.mark.parametrize(
    ("enabled", "first", "second"),
    [
        (False, "b.mlp.backward", "f.dispatch.forward"),
        (True, "f.dispatch.forward", "b.mlp.backward"),
    ],
)
def test_recompute_comm_overlap_controls_dispatch_launch_order(enabled, first, second):
    trace = []
    f_layer = _LayerPlan("f", trace, recompute_comm_overlap=enabled)
    b_layer = _LayerPlan("b", trace)

    TransformerLayerSchedulePlan.run(f_layer, b_layer, f_input=object(), b_grad=object())

    assert trace.index(first) < trace.index(second)
    # The attention recompute window remains communication-first in both modes.
    assert trace.index("f.combine.forward") < trace.index("b.attn.backward")


def _overlap_config(**kwargs):
    config = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 1,
        "bf16": True,
        "num_moe_experts": 2,
        "expert_model_parallel_size": 2,
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": True,
    }
    config.update(kwargs)
    return TransformerConfig(**config)


def test_recompute_comm_overlap_requires_selective_recompute():
    with pytest.raises(
        AssertionError, match="EP recompute communication overlap requires selective recomputation"
    ):
        _overlap_config(recompute_comm_overlap=True)


def test_recompute_comm_overlap_accepts_selective_modules():
    config = _overlap_config(
        recompute_comm_overlap=True,
        recompute_granularity="selective",
        recompute_modules=["layernorm", "moe_act", "core_attn"],
        moe_grouped_gemm=True,
    )

    assert config.recompute_comm_overlap


def test_ep_overlap_mtp_limit_does_not_depend_on_recompute_comm_overlap():
    with pytest.raises(AssertionError, match="MTP layernum only supports 1"):
        _overlap_config(mtp_num_layers=2)
