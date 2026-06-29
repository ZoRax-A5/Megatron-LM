#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generate CSV and dependency-free SVG plots from rank-local memory profile JSONL."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Iterable

BREAKDOWN_KEYS = (
    "parameter",
    "gradient",
    "optimizer",
    "saved_activation",
    "communication",
    "other_torch",
    "allocator_cache",
    "external_cuda",
    "unaccounted_overlap",
)
COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6B7280",
    "#000000",
    "#F0E442",
)
MIB = 1024 * 1024


def _rank_from_path(path: Path) -> int:
    match = re.fullmatch(r"rank(\d+)\.jsonl", path.name)
    if match is None:
        raise ValueError(f"Unexpected rank profile filename: {path}")
    return int(match.group(1))


def load_profiles(input_dir: Path) -> dict[int, list[dict]]:
    """Load rank JSONL files ordered by event index."""
    profiles = {}
    for path in sorted(input_dir.glob("rank*.jsonl"), key=_rank_from_path):
        with open(path, encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        if records:
            profiles[_rank_from_path(path)] = sorted(records, key=lambda row: row["event_index"])
    if not profiles:
        raise RuntimeError(f"No rank*.jsonl profile data found in {input_dir}")
    return profiles


def write_csv(profiles: dict[int, list[dict]], output_path: Path) -> None:
    """Write all samples to one flat CSV file."""
    fields = (
        "rank",
        "local_rank",
        "tp_rank",
        "pp_rank",
        "ep_rank",
        "dp_rank",
        "iteration",
        "event_index",
        "event",
        "microbatch",
        "vp_stage",
        "layer",
        "elapsed_ms",
        "allocated_bytes",
        "reserved_bytes",
        "interval_peak_allocated_bytes",
        "device_used_bytes",
        *BREAKDOWN_KEYS,
    )
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for records in profiles.values():
            for record in records:
                row = {key: record.get(key) for key in fields}
                row.update(record.get("breakdown_bytes", {}))
                writer.writerow(row)


def _polyline(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_line_plot(
    output_path: Path,
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    colors: dict[str, str] | None = None,
) -> None:
    """Write a simple SVG line chart with elapsed milliseconds and MiB axes."""
    width, height = 1200, 620
    left, right, top, bottom = 90, 30, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    x_span = max(max_x - min_x, 1.0)
    max_y = max(max_y, 1.0)

    def sx(value: float) -> float:
        return left + (value - min_x) / x_span * plot_width

    def sy(value: float) -> float:
        return top + plot_height - value / max_y * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for index in range(6):
        value = max_y * index / 5
        y = sy(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>'
        )
        elements.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.0f}</text>'
        )
    for index in range(6):
        value = min_x + x_span * index / 5
        x = sx(value)
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.0f}</text>'
        )

    for index, (name, points) in enumerate(series.items()):
        color = colors.get(name) if colors else COLORS[index % len(COLORS)]
        color = color or COLORS[index % len(COLORS)]
        scaled = [(sx(x), sy(y)) for x, y in points]
        elements.append(
            f'<polyline points="{_polyline(scaled)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
        legend_x = left + (index % 5) * 210
        legend_y = height - 34 + (index // 5) * 18
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>'
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="13">elapsed time (ms)</text>',
            f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">memory (MiB)</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(elements), encoding="utf-8")


def generate_plots(profiles: dict[int, list[dict]], output_dir: Path) -> None:
    """Generate per-rank and all-rank plots plus an HTML index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overview: dict[str, list[tuple[float, float]]] = {}
    links = []
    for rank, records in profiles.items():
        total_series = {
            "allocated": [(row["elapsed_ms"], row["allocated_bytes"] / MIB) for row in records],
            "reserved": [(row["elapsed_ms"], row["reserved_bytes"] / MIB) for row in records],
            "device_used": [(row["elapsed_ms"], row["device_used_bytes"] / MIB) for row in records],
            "interval_peak": [
                (row["elapsed_ms"], row["interval_peak_allocated_bytes"] / MIB) for row in records
            ],
        }
        write_line_plot(
            output_dir / f"rank{rank}_total.svg",
            title=f"Rank {rank}: total memory",
            series=total_series,
        )
        overview[f"rank{rank}"] = total_series["allocated"]

        breakdown_series = {
            key: [
                (row["elapsed_ms"], row.get("breakdown_bytes", {}).get(key, 0) / MIB)
                for row in records
            ]
            for key in BREAKDOWN_KEYS
        }
        breakdown_series = {
            key: points for key, points in breakdown_series.items() if any(y > 0 for _, y in points)
        }
        breakdown_name = None
        if breakdown_series:
            breakdown_name = f"rank{rank}_breakdown.svg"
            write_line_plot(
                output_dir / breakdown_name,
                title=f"Rank {rank}: memory breakdown",
                series=breakdown_series,
                colors={
                    key: COLORS[index % len(COLORS)] for index, key in enumerate(BREAKDOWN_KEYS)
                },
            )
        links.append((rank, f"rank{rank}_total.svg", breakdown_name))

    rank_colors = {
        name: f"hsl({index * 360 / max(len(overview), 1):.1f},70%,40%)"
        for index, name in enumerate(overview)
    }
    write_line_plot(
        output_dir / "all_ranks_allocated.svg",
        title="All ranks: allocated memory",
        series=overview,
        colors=rank_colors,
    )

    rows = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Megatron memory profile</title></head><body>'
    ]
    rows.extend(
        [
            "<h1>Megatron GPU memory profile</h1>",
            '<p><a href="../all_samples.csv">Combined CSV data</a></p>',
            '<h2>All ranks</h2><img src="all_ranks_allocated.svg" style="max-width:100%">',
            "<h2>Per-rank plots</h2><ul>",
        ]
    )
    for rank, total_name, breakdown_name in links:
        breakdown_link = f' | <a href="{breakdown_name}">breakdown</a>' if breakdown_name else ""
        rows.append(f'<li>Rank {rank}: <a href="{total_name}">total</a>{breakdown_link}</li>')
    rows.append("</ul></body></html>")
    (output_dir / "index.html").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-ranks", type=int)
    args = parser.parse_args()

    profiles = load_profiles(args.input_dir)
    if args.expected_ranks is not None and len(profiles) != args.expected_ranks:
        raise RuntimeError(
            f"Expected {args.expected_ranks} rank profiles, found {len(profiles)} in "
            f"{args.input_dir}"
        )
    write_csv(profiles, args.input_dir / "all_samples.csv")
    generate_plots(profiles, args.output_dir or args.input_dir / "plots")


if __name__ == "__main__":
    main()
