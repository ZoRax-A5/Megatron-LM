# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline Megatron parallel-layout simulator.

This tool mirrors Megatron Core's default rank group generation without
initializing torch.distributed or requiring GPUs.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class ParallelLayoutConfig:
    """Configuration for an offline Megatron parallel layout."""

    world_size: int
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int | None = None
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    gpus_per_node: int = 8
    order: str = "tp-cp-ep-dp-pp"


def _prefix_product(values: List[int], init: int = 1) -> List[int]:
    result = [init]
    for value in values:
        init *= value
        result.append(init)
    return result


def _decompose(index: int, shape: List[int], stride: List[int] | None = None) -> List[int]:
    if stride is None:
        stride = _prefix_product(shape)
    return [(index // offset) % size for size, offset in zip(shape, stride)]


def _inner_product(lhs: List[int], rhs: List[int]) -> int:
    return sum(left * right for left, right in zip(lhs, rhs))


def _generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    """Mirror megatron.core.parallel_state.generate_masked_orthogonal_rank_groups."""

    masked_shape = [size for size, is_masked in zip(parallel_size, mask) if is_masked]
    unmasked_shape = [size for size, is_masked in zip(parallel_size, mask) if not is_masked]

    global_stride = _prefix_product(parallel_size)
    masked_stride = [stride for stride, is_masked in zip(global_stride, mask) if is_masked]
    unmasked_stride = [stride for stride, is_masked in zip(global_stride, mask) if not is_masked]

    group_size = _prefix_product(masked_shape)[-1]
    num_groups = world_size // group_size

    ranks = []
    for group_index in range(num_groups):
        decomposed_group_index = _decompose(group_index, unmasked_shape)
        rank_group = []
        for rank_in_group in range(group_size):
            decomposed_rank_index = _decompose(rank_in_group, masked_shape)
            rank_group.append(
                _inner_product(decomposed_rank_index, masked_stride)
                + _inner_product(decomposed_group_index, unmasked_stride)
            )
        ranks.append(rank_group)
    return ranks


class _RankGenerator:
    """Small offline equivalent of Megatron Core's RankGenerator."""

    def __init__(
        self, *, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        if ep != 1 and cp != 1:
            raise ValueError("EP and CP cannot both be greater than 1 in one rank generator")

        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep
        self.name_to_size = {"tp": tp, "pp": pp, "dp": dp, "ep": ep, "cp": cp}

        tokens = order.lower().split("-")
        for name, size in self.name_to_size.items():
            if name not in tokens and size != 1:
                raise ValueError(
                    f"The size of ({name}) is ({size}), but it is missing from order ({order})"
                )
            if name not in tokens:
                tokens.append(name)

        self.order = "-".join(tokens)
        self.ordered_size = [self.name_to_size[token] for token in tokens]

    def get_mask(self, token: str) -> List[bool]:
        ordered_tokens = self.order.split("-")
        token_list = token.split("-")
        return [ordered_token in token_list for ordered_token in ordered_tokens]

    def get_ranks(self, token: str) -> List[List[int]]:
        ranks = _generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, self.get_mask(token)
        )
        if self.rank_offset:
            return [[rank + self.rank_offset for rank in rank_group] for rank_group in ranks]
        return ranks


def _get_first(config: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return default


def config_from_mapping(config: Mapping[str, Any]) -> ParallelLayoutConfig:
    """Build a layout config from full Megatron names or short aliases."""

    return ParallelLayoutConfig(
        world_size=int(_get_first(config, "world_size")),
        tensor_model_parallel_size=int(
            _get_first(config, "tensor_model_parallel_size", "tp", default=1)
        ),
        pipeline_model_parallel_size=int(
            _get_first(config, "pipeline_model_parallel_size", "pp", default=1)
        ),
        data_parallel_size=(
            None
            if _get_first(config, "data_parallel_size", "dp") is None
            else int(_get_first(config, "data_parallel_size", "dp"))
        ),
        context_parallel_size=int(_get_first(config, "context_parallel_size", "cp", default=1)),
        expert_model_parallel_size=int(
            _get_first(config, "expert_model_parallel_size", "ep", default=1)
        ),
        expert_tensor_parallel_size=(
            None
            if _get_first(config, "expert_tensor_parallel_size", "etp") is None
            else int(_get_first(config, "expert_tensor_parallel_size", "etp"))
        ),
        gpus_per_node=int(_get_first(config, "gpus_per_node", default=8)),
        order=str(_get_first(config, "order", default="tp-cp-ep-dp-pp")),
    )


def _load_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config files require PyYAML. Use JSON or install the test/dev dependencies."
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file {path}")
    return loaded


def _normalise_order(order: str, sizes: Mapping[str, int]) -> str:
    tokens = order.lower().split("-")
    for name in sizes:
        if name not in tokens:
            tokens.append(name)
    return "-".join(tokens)


def _rank_coordinates(rank: int, rank_generator: _RankGenerator) -> Dict[str, int]:
    sizes = dict(zip(rank_generator.order.split("-"), rank_generator.ordered_size))
    strides: Dict[str, int] = {}
    stride = 1
    for token in rank_generator.order.split("-"):
        strides[token] = stride
        stride *= sizes[token]

    local_rank = rank - rank_generator.rank_offset
    return {token: (local_rank // strides[token]) % sizes[token] for token in sizes}


def _membership(groups: Mapping[str, List[List[int]]], rank: int) -> Dict[str, int]:
    membership = {}
    for name, rank_groups in groups.items():
        for index, ranks in enumerate(rank_groups):
            if rank in ranks:
                membership[name] = index
                break
    return membership


def _build_generator(*, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> _RankGenerator:
    return _RankGenerator(tp=tp, ep=ep, dp=dp, pp=pp, cp=cp, order=order)


def _validate_config(config: ParallelLayoutConfig) -> Dict[str, int]:
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if config.gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be positive")

    tp = config.tensor_model_parallel_size
    pp = config.pipeline_model_parallel_size
    cp = config.context_parallel_size
    ep = config.expert_model_parallel_size
    etp = config.expert_tensor_parallel_size or tp

    for name, size in {"tp": tp, "pp": pp, "cp": cp, "ep": ep, "etp": etp}.items():
        if size <= 0:
            raise ValueError(f"{name} size must be positive")

    model_parallel_size = tp * pp * cp
    if config.world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size ({config.world_size}) is not divisible by tp*pp*cp "
            f"({model_parallel_size})"
        )
    dp = config.world_size // model_parallel_size
    if config.data_parallel_size is not None and config.data_parallel_size != dp:
        raise ValueError(
            f"data_parallel_size ({config.data_parallel_size}) does not match "
            f"world_size/(tp*pp*cp) ({dp})"
        )

    expert_model_parallel_size = etp * ep * pp
    if config.world_size % expert_model_parallel_size != 0:
        raise ValueError(
            f"world_size ({config.world_size}) is not divisible by etp*ep*pp "
            f"({expert_model_parallel_size})"
        )
    expert_dp = config.world_size // expert_model_parallel_size

    order = _normalise_order(config.order, {"tp": tp, "cp": cp, "ep": ep, "dp": dp, "pp": pp})
    if not order.endswith("pp") and pp != 1 and expert_dp != dp:
        raise ValueError(
            "When order is not pp-last, Megatron requires attention and MoE data "
            "parallel sizes to match"
        )

    return {"tp": tp, "pp": pp, "cp": cp, "ep": ep, "etp": etp, "dp": dp, "expert_dp": expert_dp}


def generate_layout(config: ParallelLayoutConfig) -> Dict[str, Any]:
    """Generate Megatron-style rank groups and per-rank layout metadata."""

    sizes = _validate_config(config)
    decoder_generator = _build_generator(
        tp=sizes["tp"], ep=1, dp=sizes["dp"], pp=sizes["pp"], cp=sizes["cp"], order=config.order
    )
    expert_generator = _build_generator(
        tp=sizes["etp"],
        ep=sizes["ep"],
        dp=sizes["expert_dp"],
        pp=sizes["pp"],
        cp=1,
        order=config.order,
    )

    decoder_groups = {
        "tp": decoder_generator.get_ranks("tp"),
        "pp": decoder_generator.get_ranks("pp"),
        "dp": decoder_generator.get_ranks("dp"),
        "cp": decoder_generator.get_ranks("cp"),
        "tp_cp": decoder_generator.get_ranks("tp-cp"),
        "dp_cp": decoder_generator.get_ranks("dp-cp"),
        "tp_dp": decoder_generator.get_ranks("tp-dp"),
        "model": decoder_generator.get_ranks("tp-pp-cp"),
    }
    expert_groups = {
        "expert_tp": expert_generator.get_ranks("tp"),
        "expert_pp": expert_generator.get_ranks("pp"),
        "expert_dp": expert_generator.get_ranks("dp"),
        "expert_ep": expert_generator.get_ranks("ep"),
        "expert_tp_ep": expert_generator.get_ranks("tp-ep"),
        "expert_tp_ep_pp": expert_generator.get_ranks("tp-ep-pp"),
    }

    ranks = []
    for rank in range(config.world_size):
        decoder_coordinates = _rank_coordinates(rank, decoder_generator)
        expert_coordinates = _rank_coordinates(rank, expert_generator)
        ranks.append(
            {
                "rank": rank,
                "node": rank // config.gpus_per_node,
                "local_gpu": rank % config.gpus_per_node,
                "decoder_coordinates": {
                    "tp": decoder_coordinates["tp"],
                    "pp": decoder_coordinates["pp"],
                    "dp": decoder_coordinates["dp"],
                    "cp": decoder_coordinates["cp"],
                },
                "expert_coordinates": {
                    "expert_tp": expert_coordinates["tp"],
                    "pp": expert_coordinates["pp"],
                    "expert_dp": expert_coordinates["dp"],
                    "expert_ep": expert_coordinates["ep"],
                },
                "decoder_group_ids": _membership(decoder_groups, rank),
                "expert_group_ids": _membership(expert_groups, rank),
            }
        )

    return {
        "metadata": {
            "config": asdict(config),
            "resolved_sizes": sizes,
            "decoder_order": decoder_generator.order,
            "expert_order": expert_generator.order,
            "hardware_assumption": {
                "gpus_per_node": config.gpus_per_node,
                "intra_node": "NVLink",
                "inter_node": "InfiniBand",
                "rank_to_gpu": "rank // gpus_per_node is node; rank % gpus_per_node is local GPU",
            },
        },
        "groups": {"decoder": decoder_groups, "expert": expert_groups},
        "ranks": ranks,
    }


def _ranks_for_cell(
    ranks: List[Mapping[str, Any]],
    *,
    coordinate_key: str,
    dp_key: str,
    pp_key: str,
    dp_rank: int,
    pp_rank: int,
) -> List[Mapping[str, Any]]:
    return [
        rank
        for rank in ranks
        if rank[coordinate_key][dp_key] == dp_rank and rank[coordinate_key][pp_key] == pp_rank
    ]


def _format_decoder_cell(cell_ranks: List[Mapping[str, Any]]) -> str:
    values = []
    for rank in sorted(cell_ranks, key=lambda item: item["rank"]):
        coords = rank["decoder_coordinates"]
        values.append(
            f"r{rank['rank']}(n{rank['node']}g{rank['local_gpu']},"
            f"tp{coords['tp']},cp{coords['cp']})"
        )
    return "<br>".join(values) if values else "-"


def _format_expert_cell(cell_ranks: List[Mapping[str, Any]]) -> str:
    values = []
    for rank in sorted(cell_ranks, key=lambda item: item["rank"]):
        coords = rank["expert_coordinates"]
        values.append(
            f"r{rank['rank']}(n{rank['node']}g{rank['local_gpu']},"
            f"etp{coords['expert_tp']},ep{coords['expert_ep']})"
        )
    return "<br>".join(values) if values else "-"


def render_markdown(layout: Mapping[str, Any]) -> str:
    """Render a compact Markdown layout report."""

    metadata = layout["metadata"]
    sizes = metadata["resolved_sizes"]
    ranks = layout["ranks"]
    world_size = metadata["config"]["world_size"]
    gpus_per_node = metadata["config"]["gpus_per_node"]
    node_count = (world_size + gpus_per_node - 1) // gpus_per_node

    lines = [
        "# Megatron Parallel Layout",
        "",
        "## Summary",
        "",
        f"- World size: {world_size}",
        f"- Nodes: {node_count} x {gpus_per_node} GPUs",
        "- Hardware assumption: NVLink inside a node, InfiniBand across nodes",
        f"- Order: `{metadata['decoder_order']}`",
        (
            f"- Decoder sizes: TP={sizes['tp']}, PP={sizes['pp']}, "
            f"DP={sizes['dp']}, CP={sizes['cp']}"
        ),
        (
            f"- Expert sizes: ETP={sizes['etp']}, PP={sizes['pp']}, "
            f"Expert-DP={sizes['expert_dp']}, EP={sizes['ep']}"
        ),
        "",
        "## Decoder Layout",
        "",
        "Rows are DP ranks; columns are PP ranks. Each cell lists logical ranks and "
        "`node/local_gpu`, TP, and CP coordinates.",
        "",
    ]

    header = ["DP \\ PP"] + [f"PP {pp_rank}" for pp_rank in range(sizes["pp"])]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for dp_rank in range(sizes["dp"]):
        row = [f"DP {dp_rank}"]
        for pp_rank in range(sizes["pp"]):
            cell_ranks = _ranks_for_cell(
                ranks,
                coordinate_key="decoder_coordinates",
                dp_key="dp",
                pp_key="pp",
                dp_rank=dp_rank,
                pp_rank=pp_rank,
            )
            row.append(_format_decoder_cell(cell_ranks))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Expert Layout",
            "",
            "Rows are expert-DP ranks; columns are PP ranks. Each cell lists logical ranks and "
            "`node/local_gpu`, expert TP, and EP coordinates.",
            "",
        ]
    )
    header = ["Expert-DP \\ PP"] + [f"PP {pp_rank}" for pp_rank in range(sizes["pp"])]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for expert_dp_rank in range(sizes["expert_dp"]):
        row = [f"Expert-DP {expert_dp_rank}"]
        for pp_rank in range(sizes["pp"]):
            cell_ranks = _ranks_for_cell(
                ranks,
                coordinate_key="expert_coordinates",
                dp_key="expert_dp",
                pp_key="pp",
                dp_rank=expert_dp_rank,
                pp_rank=pp_rank,
            )
            row.append(_format_expert_cell(cell_ranks))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Node Map", ""])
    lines.append("| Node | Ranks |")
    lines.append("| --- | --- |")
    for node in range(node_count):
        node_ranks = [
            f"r{rank['rank']}=g{rank['local_gpu']}" for rank in ranks if rank["node"] == node
        ]
        lines.append(f"| Node {node} | {' '.join(node_ranks)} |")

    return "\n".join(lines) + "\n"


def write_outputs(layout: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parallel_groups.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "layout.md").write_text(render_markdown(layout), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON or YAML config file")
    parser.add_argument("--output-dir", type=Path, default=Path("parallel_layout_output"))
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--tp", "--tensor-model-parallel-size", dest="tp", type=int)
    parser.add_argument("--pp", "--pipeline-model-parallel-size", dest="pp", type=int)
    parser.add_argument("--dp", "--data-parallel-size", dest="dp", type=int)
    parser.add_argument("--cp", "--context-parallel-size", dest="cp", type=int)
    parser.add_argument("--ep", "--expert-model-parallel-size", dest="ep", type=int)
    parser.add_argument("--etp", "--expert-tensor-parallel-size", dest="etp", type=int)
    parser.add_argument("--order", type=str)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ParallelLayoutConfig:
    values: Dict[str, Any] = {}
    if args.config is not None:
        values.update(_load_config_file(args.config))

    overrides = {
        "world_size": args.world_size,
        "gpus_per_node": args.gpus_per_node,
        "tp": args.tp,
        "pp": args.pp,
        "dp": args.dp,
        "cp": args.cp,
        "ep": args.ep,
        "etp": args.etp,
        "order": args.order,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if "world_size" not in values:
        raise ValueError("world_size is required via --world-size or config file")
    return config_from_mapping(values)


def main() -> None:
    args = _parse_args()
    layout = generate_layout(_config_from_args(args))
    write_outputs(layout, args.output_dir)
    print(f"Wrote {args.output_dir / 'parallel_groups.json'}")
    print(f"Wrote {args.output_dir / 'layout.md'}")


if __name__ == "__main__":
    main()
