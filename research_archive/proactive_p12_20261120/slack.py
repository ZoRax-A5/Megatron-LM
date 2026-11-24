"""Evaluate fixed-trace holes on the compute stream occupied by an inserted Replay."""
from pathlib import Path
import json
import statistics
root=Path(__file__).resolve().parent
base=root/"p12_4b_C0_trace_conn1"
ref=json.loads((base/"nvtx_attribution.json").read_text())
tasks=json.loads((base/"tasks.json").read_text())

def traffic_slots(begin,end,intervals):
    res=[]
    cursor=begin
    for a,b in sorted(intervals):
        if a>cursor:
            res.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<end:
        res.append((cursor,end))
    return res

out={}
for mode, bucket_idx in (("C1",35),("Clate",34)):
    marks=json.loads((root/f"p12_4b_{mode}_trace_conn1"/"nvtx_attribution.json").read_text())
    ranks=[]
    for rank in range(8):
        key=str(rank)
        inserted,=[e for e in marks[key] if e["kind"]=="qkv" and e["layer"]==1
                   and e["stage"]==["prefetch",7]]
        streams={k["stream"] for k in inserted["kernels"]}
        assert len(streams)==1
        trace_step,=[e for e in ref[key] if e["kind"]=="step"]
        target,=[r for r in tasks if r["mb"]==7 and r["rank"]==rank and r["layer"]==1]
        host=sorted((m for m in ref[key] if m["kind"]=="grad_sync"),key=lambda e:e["cpu_start"])[bucket_idx]
        rs,=[k for k in host["kernels"] if "nccl" in k["name"].lower()]
        lo=max(rs["start"],host["cpu_end"],target["ready_upper_ns"])
        hi=min(rs["end"],target["deadline_ns"])
        assert lo<hi
        blocked=[(max(lo,k["start"]),min(hi,k["end"])) for k in trace_step["kernels"]
                 if k["stream"] in streams and max(lo,k["start"])<min(hi,k["end"])]
        slots=traffic_slots(lo,hi,blocked)
        unions=sum(b-a for a,b in slots)
        maxhole=max((b-a for a,b in slots),default=0)
        replay_cost=sum(k["end"]-k["start"] for k in inserted["kernels"])
        ranks.append(dict(rank=rank, window_ms=(hi-lo)/1e6, stream_open_total_ms=unions/1e6,
                          max_hole_ms=maxhole/1e6, fits_actual_replay=maxhole>=replay_cost,
                          fits_c0_replay=maxhole/1e6>=target["replay_gpu_ms"]))
    out[mode]=ranks
(root/"fixed_trace_slack.json").write_text(json.dumps(out,indent=2))
for m,r in out.items():
    print(m, "actual replay fits", sum(x["fits_actual_replay"] for x in r),
          "largest holes",[(k,statistics.median(x[k] for x in r))
                           for k in ("window_ms","max_hole_ms","stream_open_total_ms")])
