# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.parallel_layout import (
    ParallelLayoutConfig,
    config_from_mapping,
    generate_layout,
    render_markdown,
    write_outputs,
)


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            ParallelLayoutConfig(
                world_size=8, tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            ),
            {"tp": 1, "pp": 1, "dp": 8, "cp": 1, "ep": 1, "etp": 1, "expert_dp": 8},
        ),
        (
            ParallelLayoutConfig(
                world_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2
            ),
            {"tp": 2, "pp": 2, "dp": 4, "cp": 1, "ep": 1, "etp": 2, "expert_dp": 4},
        ),
        (
            ParallelLayoutConfig(
                world_size=64,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=8,
                data_parallel_size=8,
            ),
            {"tp": 1, "pp": 8, "dp": 8, "cp": 1, "ep": 1, "etp": 1, "expert_dp": 8},
        ),
        (
            ParallelLayoutConfig(
                world_size=128,
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=4,
                context_parallel_size=2,
                expert_model_parallel_size=4,
                expert_tensor_parallel_size=1,
            ),
            {"tp": 2, "pp": 4, "dp": 8, "cp": 2, "ep": 4, "etp": 1, "expert_dp": 8},
        ),
    ],
)
def test_generate_layout_resolved_sizes(config, expected):
    layout = generate_layout(config)

    assert layout["metadata"]["resolved_sizes"] == expected
    assert len(layout["ranks"]) == config.world_size
    assert layout["ranks"][0]["node"] == 0
    assert layout["ranks"][0]["local_gpu"] == 0
    assert (
        layout["ranks"][min(8, config.world_size - 1)]["node"]
        == min(8, config.world_size - 1) // config.gpus_per_node
    )


def test_config_from_mapping_accepts_short_aliases():
    config = config_from_mapping(
        {
            "world_size": 64,
            "tp": 1,
            "pp": 8,
            "dp": 8,
            "cp": 1,
            "ep": 2,
            "etp": 1,
            "gpus_per_node": 8,
        }
    )

    assert config == ParallelLayoutConfig(
        world_size=64,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=8,
        data_parallel_size=8,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        gpus_per_node=8,
    )


def test_layout_uses_megatron_default_grouping_for_8_gpu_pp_dp():
    layout = generate_layout(
        ParallelLayoutConfig(
            world_size=8,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=4,
        )
    )

    assert layout["groups"]["decoder"]["pp"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert layout["groups"]["decoder"]["dp"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert layout["ranks"][5]["decoder_coordinates"] == {"tp": 0, "pp": 1, "dp": 1, "cp": 0}


def test_layout_supports_separate_expert_tensor_parallel_size():
    layout = generate_layout(
        ParallelLayoutConfig(
            world_size=16,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        )
    )

    assert layout["metadata"]["resolved_sizes"]["dp"] == 4
    assert layout["metadata"]["resolved_sizes"]["expert_dp"] == 4
    assert layout["groups"]["expert"]["expert_ep"][0] == [0, 1]
    assert layout["groups"]["expert"]["expert_tp"][0] == [0]


def test_write_outputs_creates_json_and_markdown(tmp_path):
    layout = generate_layout(
        ParallelLayoutConfig(
            world_size=8, tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
    )

    write_outputs(layout, tmp_path)

    json_path = tmp_path / "parallel_groups.json"
    markdown_path = tmp_path / "layout.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["metadata"]["config"]["world_size"] == 8
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Megatron Parallel Layout" in markdown
    assert "## Decoder Layout" in markdown
    assert "## Expert Layout" in markdown


def test_render_markdown_has_pp_columns_and_node_map():
    layout = generate_layout(
        ParallelLayoutConfig(
            world_size=16, tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
    )

    markdown = render_markdown(layout)

    assert "| DP \\ PP | PP 0 | PP 1 |" in markdown
    assert "| Node 0 | r0=g0 r1=g1 r2=g2 r3=g3 r4=g4 r5=g5 r6=g6 r7=g7 |" in markdown
    assert "| Node 1 | r8=g0 r9=g1 r10=g2 r11=g3 r12=g4 r13=g5 r14=g6 r15=g7 |" in markdown


def test_rejects_inconsistent_data_parallel_size():
    with pytest.raises(ValueError, match="data_parallel_size"):
        generate_layout(
            ParallelLayoutConfig(
                world_size=64,
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=4,
                data_parallel_size=2,
            )
        )
