#!/usr/bin/env python3
"""Summarize full-tensor P17 C0-to-variant correctness JSON files."""
import collections
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
pattern = sys.argv[2]
files = sorted(root.glob(pattern))
totals = collections.defaultdict(lambda: {"tensors": 0, "elements": 0, "different": 0, "violations": 0, "nonfinite": 0})
steps = []
ranks = set()
all_equal = True
for path in files:
    steps.append(path.stem.split("step")[-1])
    data = json.loads(path.read_text())
    for row in data:
        ranks.add(row["rank"])
        for phase in ("initial", "observed"):
            for name, metrics in row.get(phase, {}).get("sections", {}).items():
                out = totals[f"{phase}.{name}"]
                for key in out:
                    out[key] += metrics.get(key, 0)
                all_equal &= metrics.get("different", 0) == 0 and metrics.get("violations", 0) == 0 and metrics.get("nonfinite", 0) == 0
summary = {
    "files": [str(p) for p in files],
    "steps": steps,
    "rank_count": len(ranks),
    "ranks": sorted(ranks),
    "all_elements_equal": all_equal,
    "sections": dict(totals),
}
json.dump(summary, sys.stdout, indent=2)
print()
