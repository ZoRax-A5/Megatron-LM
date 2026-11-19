# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validation for zero-based global layer recomputation settings."""
import dataclasses
import json

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


def config(mapping, **kwargs):
    return TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        recompute_granularity='selective',
        recompute_layer_config=mapping,
        **kwargs,
    )


def test_normalization_and_union():
    result = config({'0': ['attn_qkv'], 1: ['mlp_fc1'], '2': []})
    assert result.recompute_layer_config == {0: ['attn_qkv'], 1: ['mlp_fc1'], 2: []}
    assert set(result.recompute_modules) == {'attn_qkv', 'mlp_fc1'}


def test_empty_and_different_layers_can_use_whole_and_fine_mlp():
    result = config({'0': ['mlp'], '1': ['mlp_fc1']})
    assert result.recompute_layer_config[1] == ['mlp_fc1']
    assert config({}).recompute_modules == []


@pytest.mark.parametrize(
    'mapping',
    [
        {'4': ['mlp_fc1']},
        {'-1': []},
        {True: []},
        {'0': ['unknown']},
        {'1': 'mlp_fc1'},
        {'01': [], 1: []},
        {'2': ['mlp', 'mlp_fc1']},
        {'0': ['attn_qkv', 'attn_qkv']},
        [],
    ],
)
def test_bad_mapping_rejected(mapping):
    with pytest.raises((ValueError, AssertionError)):
        config(mapping)


def test_requires_selective():
    with pytest.raises(ValueError, match='requires selective'):
        TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            recompute_granularity=None,
            recompute_layer_config={},
        )


def test_cli_json_parsing():
    item = next(
        f for f in dataclasses.fields(TransformerConfig) if f.name == 'recompute_layer_config'
    )
    opts = ArgumentGroupFactory(TransformerConfig)._build_argparse_kwargs_from_field(item)
    assert opts['type']('{"0": ["attn_qkv"]}') == {'0': ['attn_qkv']}


def test_cli_rejects_duplicate_json_keys():
    item = next(
        f for f in dataclasses.fields(TransformerConfig) if f.name == "recompute_layer_config"
    )
    opts = ArgumentGroupFactory(TransformerConfig)._build_argparse_kwargs_from_field(item)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        opts["type"]('{"0": [], "0": ["attn_qkv"]}')


def test_moe_layerwise_distinct_expert_boundaries():
    result = config({'0': ['moe_fc1'], '1': ['moe_act', 'moe_fc2'], '2': ['moe']},
                    num_moe_experts=4, moe_grouped_gemm=True, add_bias_linear=False)
    assert set(result.recompute_modules) == {'moe_fc1', 'moe_act', 'moe_fc2', 'moe'}


@pytest.mark.parametrize('mapping,kwargs', [
    ({'0': ['moe', 'moe_fc1']}, {}),
    ({'0': ['moe_fc1']}, {'add_bias_linear': True}),
    ({'0': ['moe_fc2']}, {'moe_grouped_gemm': False}),
    ({'0': ['moe_act']}, {'use_transformer_engine_op_fuser': True}),
])
def test_moe_unsupported_combinations_rejected(mapping, kwargs):
    options = dict(num_moe_experts=4, moe_grouped_gemm=True, add_bias_linear=False)
    options.update(kwargs)
    with pytest.raises((ValueError, AssertionError)):
        config(mapping, **options)
