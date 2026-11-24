"""Extract per-rank replay deadlines and dispatch windows from native C0."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
case = root/"p11_4b_S0_trace_conn1"
tasks = json.loads((case/"tasks.json").read_text())
marks = json.loads((case/"nvtx_attribution.json").read_text())
rows = []
for task in tasks:
    if task["mb"] != 7:
        continue
    rank = task["rank"]
    dispatch = sorted((x for x in marks[str(rank)] if x["kind"]=="grad_sync"),key=lambda e:e["cpu_start"])
    for window in task["dp_candidates"]:
        i = window["bucket"]
        rows.append(dict(task, bucket=i, rs_window_ms=window["left_ms"],
                         post_submit_cpu_to_replay_gpu_ms=(task["deadline_ns"]-dispatch[i]["cpu_end"])/1e6))
(root/"candidate_windows.json").write_text(json.dumps(rows,indent=2))
for layer in (1,2,18):
    for i in (0,18,34,35):
        x = [r for r in rows if r["layer"]==layer and r["bucket"]==i]
        if len(x)==8:
            print(layer, i, min(r["rs_window_ms"] for r in x),
                  statistics.median(r["post_submit_cpu_to_replay_gpu_ms"] for r in x))
