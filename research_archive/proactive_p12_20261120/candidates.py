"""Bound the gain for a small multi-Replay set and find matched TP candidate slots."""
import json
from pathlib import Path
import statistics as st
import sys

case = Path(sys.argv[1])
out = Path(__file__).parent
rows = json.loads((case/"nvtx_attribution.json").read_text())
tasks = json.loads((case/"tasks.json").read_text())
comm = json.loads((case/"comm.json").read_text())
exposure = json.loads((out/(case.name+"_exposure.json")).read_text())
def summary(values):
    v = list(values)
    return dict(n=len(v), min=min(v), median=st.median(v), max=max(v))
def kid(k):
    return tuple(k[c] for c in ("pid","correlation","start","end","stream"))

layer_tasks = {(r["rank"],r["layer"]):r for r in tasks if r["mb"]==7}
candidates = {}
for layer in (1,2,18):
    ranks = []
    for rank in range(8):
        task = layer_tasks[rank,layer]
        mark = sorted([r for r in rows[str(rank)] if r["kind"]=="grad_sync"],key=lambda r:r["cpu_start"])
        eligible = [x for x in task["dp_candidates"] if x["left_ms"]>=task["replay_gpu_ms"]]
        last = max(eligible,key=lambda x:x["bucket"])
        ranks.append(dict(rank=rank,ready_upper_ns=task["ready_upper_ns"],deadline_ns=task["deadline_ns"],
            ready_to_bwd_ms=(task["deadline_ns"]-task["ready_upper_ns"])/1e6,
            replay_kernel_ms=task["replay_gpu_ms"],existing_dp_kernel_overlap_ms=task["dp_actual_ms"],
            last_bucket=last["bucket"],window_after_return_ms=last["left_ms"],
            last_lead_ms=(task["deadline_ns"]-mark[last["bucket"]]["cpu_end"])/1e6))
    candidates[layer] = ranks

# Use measured group membership (P12) or TP2's contiguous rank layout (P11).
model_ar = {kid(k):k for k in comm if k["kind"]=="tp_model" and k["operation"]=="allreduce"}
ar = {}
for rank in range(8):
    entries=[]
    for mark in rows[str(rank)]:
        if mark["kind"]!="all_reduce_api":
            continue
        for k in mark["kernels"]:
            if kid(k) in model_ar:
                peers=tuple(sorted(mark.get("peers",[rank//2*2,rank//2*2+1])))
                if len(peers)==2:
                    entries.append(dict(stage=mark["stage"],peers=peers,cpu_end=mark["cpu_end"],gpu=k))
    ar[rank]=sorted(entries,key=lambda k:k["cpu_end"])

tp_candidates=[]
for rank in (0,2,4,6):
    aa,bb=ar[rank],ar[rank+1]
    assert len(aa)==len(bb)
    assert [(a["stage"],a["gpu"]["name"]) for a in aa]==[(b["stage"],b["gpu"]["name"]) for b in bb]
    for i,(a,b) in enumerate(zip(aa,bb)):
        if a["stage"] != ["backward",7]:
            continue
        for layer in (1,2,18):
            ts=[layer_tasks[rank,layer],layer_tasks[rank+1,layer]]
            lo=max(a["cpu_end"],b["cpu_end"],a["gpu"]["start"],b["gpu"]["start"],
                   *(x["ready_upper_ns"] for x in ts))
            hi=min(a["gpu"]["end"],b["gpu"]["end"],*(x["deadline_ns"] for x in ts))
            if hi>lo:
                tp_candidates.append(dict(tp_group=list(a["peers"]),collective_ordinal=i,layer=layer,
                    gpu_kernel_ns=[a["gpu"]["end"]-a["gpu"]["start"],b["gpu"]["end"]-b["gpu"]["start"]],
                    window_ms=(hi-lo)/1e6,
                    complete_baseline_replay_fits=(hi-lo)/1e6>=max(x["replay_gpu_ms"] for x in ts)))

label = "P12 measured peers" if any("peers" in r for rr in rows.values() for r in rr) else "P11 contiguous TP pairs"
bound = [sum(layer_tasks[rank,l]["replay_gpu_ms"] for l in (1,2,18)) for rank in range(8)]
full=dict(layer_candidates=candidates,
          optimistic_per_rank_saved_replay_kernel_ms=summary(bound),
          assumed_extra_graph_mib_each=22.0078125,max_two_graphs_nominal_mib=44.015625,
          tp_pair_source=label,tp_candidates=tp_candidates,
          tp_kernel_summary=summary(r["tp_final_bwd_ar_kernel_us"]["median"] for r in exposure["ranks"]))
(out/(case.name+"_candidates.json")).write_text(json.dumps(full,indent=2))
for l,v in candidates.items():
    print(l,"buckets",summary(x["last_bucket"] for x in v),
          "lead",summary(x["last_lead_ms"] for x in v),
          "existing overlap",summary(x["existing_dp_kernel_overlap_ms"] for x in v))
print("optimistic math upper",summary(bound))
print("TP candidates",len(tp_candidates), "fit",sum(x["complete_baseline_replay_fits"] for x in tp_candidates),
      "max",max((x["window_ms"] for x in tp_candidates),default=0))
