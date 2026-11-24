"""Validate matched GPU schedule and extract Replay residency and API spans."""
import json
from pathlib import Path
import statistics as st

root = Path(__file__).resolve().parent
A={}
refs={}
native = [[phase,mb] for mb in range(8) for phase in ("forward","backward")]

def stat(xs):
    return dict(min=min(xs),median=st.median(xs),max=max(xs))

for mode in ("C0","Cnoop","C1","Clate"):
    path=root/f"p12_4b_{mode}_trace_conn1"
    marks=json.loads((path/"nvtx_attribution.json").read_text())
    comm=json.loads((path/"comm.json").read_text())
    ranks=[]
    for rank in range(8):
        row=json.loads((path/f"rank{rank}.json").read_text())["steps"][-1]
        assert row["qkv_calls"]==312 and row["schedule"]==native
        key=(rank,"dp_bucket_order")
        dp=[e["group"] for e in row["sync"]]
        assert len(dp)==37 and dp==refs.setdefault(key,dp)
        assert all(x["stage"]==["backward",7] and x["async_op"] for x in row["reduce_scatter"])
        mb_rows=sorted([x for x in marks[str(rank)] if x["kind"] in ("forward","backward")],key=lambda e:e["cpu_start"])
        comm_map={(c["start"],c["pid"],c["correlation"]):c["name"] for c in comm if
                  c["kind"]=="tp_model" and c["rank"]==rank}
        tp=[]
        for e in mb_rows:
            tp.append([comm_map[k["start"],k["pid"],k["correlation"]]
                       for k in sorted(e["kernels"],key=lambda k:k["correlation"])
                       if (k["start"],k["pid"],k["correlation"]) in comm_map])
        key=(rank,"tp_kernel_order")
        assert tp==refs.setdefault(key,tp)
        rank_data=dict(rank=rank, trigger_count=row["trigger_count"],hits=row["hits"])
        assert row["trigger_count"]==int(mode!="C0") and row["hits"]==int(mode in ("C1","Clate"))
        if mode=="Cnoop":
            noop,=[e for e in marks[str(rank)] if e["kind"]=="noop"]
            assert not noop["kernels"]
            rank_data["noop_range_us"]=(noop["cpu_end"]-noop["cpu_start"])/1e3
        if mode in ("C1","Clate"):
            cp,=[v for v in row["checkpoints"] if v["hit"]]
            cid=cp["id"]
            times={n:t for n,_,t in row["timeline"]}
            ready=times[f"replay:{cid}:7:prefetch_complete"]
            consume=times[f"consume:{cid}"]
            release=times[f"release:{cid}"]
            assert ready<consume<release
            ev={e["event"]:e for e in cp["events"]}
            assert ev["prefetch_start"]["trigger_site"] in (
                "dp_rs_dispatch:34","dp_rs_dispatch:35")
            assert cp["final_state"]=="consumed" and "fallback" not in ev
            assert not any(e["kind"]=="qkv" and e["stage"]==["backward",7] and e["layer"]==1
                           for e in marks[str(rank)])
            mem=ev["released"]["memory"]
            rank_data.update(lifetime_gpu_ms=consume-ready,
                lifetime_cpu_ms=ev["released"]["cache_lifetime_seconds"]*1000,
                cache_delta_bytes=mem["allocated_after"]-mem["allocated_before"])
        ranks.append(rank_data)
    A[mode]=dict(ranks=ranks,summary={k:stat([x[k] for x in ranks]) for k in ranks[0] if k!="rank"})
(root/"timeline_audit.json").write_text(json.dumps(A,indent=2))
print(json.dumps({k:v["summary"] for k,v in A.items()},indent=2))
