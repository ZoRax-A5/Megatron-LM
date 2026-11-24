"""Summarize full-rank P9 data without substituting missing cases."""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
report = {}
for suite in sorted(root.iterdir()):
    if not suite.is_dir():
        continue
    modes = {}
    for case in sorted(suite.glob("S*")):
        ranks = {path.stem: json.loads(path.read_text())["steps"]
                 for path in case.glob("rank*.json")}
        if not ranks or not all(ranks.values()):
            continue
        durations = [float(value) for value in re.findall(
            r"elapsed time per iteration \(ms\): ([\d.]+)", (case/"train.log").read_text())]
        rows = [step for rank in ranks.values() for step in rank]
        hits = [h for row in rows for h in row["checkpoints"] if h["hit"]]
        life = [e["cache_lifetime_seconds"]*1000 for row in hits for e in row["events"]
                if e["event"] == "released"]
        lead = []
        for step in rows:
            gpu = {name: elapsed for name, _, elapsed in step.get("timeline", [])}
            if gpu:
                for h in step["checkpoints"]:
                    if h["hit"]:
                        mb, cid = h["metadata"]["microbatch"], h["id"]
                        lead.append(gpu[f"consume:{cid}"] - gpu[f"replay:{cid}:{mb+1}:prefetch_complete"])
        if hits:
            identities_by_rank = []
            for rank, values in sorted(ranks.items()):
                identity = []
                for step in values:
                    for checkpoint in step["checkpoints"]:
                        meta = checkpoint["metadata"]
                        assert meta["rank"] == int(rank[4:]) and meta["step"] == step["iteration"]
                        assert meta["layer_number"] in (1, 2) and meta["module"] == "attn_qkv"
                        identity.append((checkpoint["id"], meta["step"], meta["microbatch"],
                                         meta["layer_number"], checkpoint["hit"]))
                assert len({row[0] for row in identity}) == len(identity)
                identities_by_rank.append(identity)
            assert all(rows == identities_by_rank[0] for rows in identities_by_rank)
        checks = {p.stem: json.loads(p.read_text())["checks"]
                  for p in sorted(case.glob("verification_rank*.json"))}
        def distribution(values):
            if not values:
                return None
            return dict(n=len(values), mean=statistics.mean(values),
                        sd=statistics.stdev(values) if len(values)>1 else 0,
                        min=min(values), median=statistics.median(values), max=max(values))
        modes[case.name] = dict(
            rank_count=len(ranks), steps=len(rows), verified_by_rank={
                key: dict(count=len(value), max_abs=max((x["max_abs"] for x in value), default=None),
                          bad=sum(x["bad_elements"] for x in value)) for key, value in checks.items()},
            memory={rank: {key: max(step[key] for step in values)
                           for key in ("allocated", "reserved")}
                    for rank, values in ranks.items()},
            allocated_steady_trend={rank: (
                max(step["allocated"] for step in values[-5:]) -
                max(step["allocated"] for step in values[5:10]))
                for rank, values in ranks.items() if len(values)>=15},
            calls=sum(row["qkv_calls"] for row in rows),
            hit_count=len(hits), released=sum(row["released"] for row in rows),
            max_cached=max(row.get("max_cached",0) for row in rows),
            rs_calls=sum(len(row.get("reduce_scatter", [])) for row in rows),
            rs_per_step_counts=dict(Counter(len(row.get("reduce_scatter", [])) for row in rows)),
            cache_cpu_ms=distribution(life), replay_to_consume_gpu_ms=distribution(lead),
            iteration_ms=distribution(durations), steady_ms=distribution(durations[5:])
        )
    if modes:
        report[suite.name] = modes
(root/"summary.json").write_text(json.dumps(report, indent=2))
