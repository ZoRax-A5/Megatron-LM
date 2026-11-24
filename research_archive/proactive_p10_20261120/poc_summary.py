"""Aggregate targeted kernel intersections, event handoffs, and unprofiled step data."""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
modes = ("S0","S1","S2","same","side")

def stats(xs):
    return dict(n=len(xs), min=min(xs), median=statistics.median(xs), max=max(xs)) if xs else None

reports = {"timeline": {}, "perf": {}}
for mode in ("same","side"):
    case = root/f"poc_4b_{mode}_trace_conn1_fenced"
    if not (case/"tasks.json").exists():
        continue
    marks = json.loads((case/"nvtx_attribution.json").read_text())
    tasks = json.loads((case/"tasks.json").read_text())
    comm = json.loads((case/"comm.json").read_text())
    report = []
    for rank in range(8):
        target, = [row for row in tasks if row["rank"] == rank and row["stage"] == "prefetch"
                   and row["mb"] == 7 and row["layer"] == 1]
        assert not any(row["rank"]==rank and row["stage"]=="backward"
                       and row["mb"]==7 and row["layer"]==1 for row in tasks)
        r = json.loads((case/f"rank{rank}.json").read_text())["steps"][-1]
        cp, = [x for x in r["checkpoints"] if x["id"]==r["target_id"]]
        events = {e["event"]:e for e in cp["events"]}
        assert cp["hit"] and cp["final_state"]=="consumed" and r["target_consumed"]
        cpu_event_spans = {e["kind"]: (e["cpu_end"]-e["cpu_start"])/1e3
                           for e in marks[str(rank)] if "_wait" in e["kind"]}
        timeline = {n:ms for n,_,ms in r["timeline"]}
        replay_prefix = f"replay:{cp['id']}:7"
        first_rs = min((x for x in comm if x["rank"]==rank and x["kind"]=="grad_rs"),
                       key=lambda x:x["start"])
        compute = [x for x in marks[str(rank)] if x["kind"]=="qkv"
                   and x["stage"]==["prefetch",7] and x["layer"]==1][0]["kernels"]
        intersection = sum(max(0,min(x["end"],first_rs["end"])-max(x["start"],first_rs["start"]))
                           for x in compute)/1e6
        report.append(dict(rank=rank, overlap_ms=target["dp_actual_ms"],
                           first_bucket_overlap_ms=intersection, replay_ms=target["replay_gpu_ms"],
                           first_rs_ms=(first_rs["end"]-first_rs["start"])/1e6,
                           replay_streams=target["compute_streams"], rs_stream=first_rs["stream"],
                           resident_delta=r["target_allocated_delta"], output_bytes=events["prefetch_complete"]["output_bytes"],
                           target_lifetime_cpu_ms=events["released"]["cache_lifetime_seconds"]*1000,
                           replay_to_consume_event_ms=(timeline[f"consume:{cp['id']}"]-
                               timeline[replay_prefix+":end"]), wait_cpu_us=cpu_event_spans))
    reports["timeline"][mode] = report

for mode, rep in [(mode,"final") for mode in modes] + [
    ("same","r2"),("side","r2"),("same","r3"),("side","r3")
]:
    case = root/f"poc_4b_{mode}_perf_conn1_{rep}"
    if not all((case/f"rank{rank}.json").exists() for rank in range(8)):
        continue
    log = (case/"train.log").read_text()
    times = [float(t) for t in re.findall(
        r"elapsed time per iteration \(ms\): ([\d.]+)",log)]
    assert len(times)==12 and not re.search(r"skipped iterations:\s+[1-9]|nan iterations:\s+[1-9]",log)
    rank_rows = []
    for rank in range(8):
        data = json.loads((case/f"rank{rank}.json").read_text())["steps"]
        assert len(data)==12
        for x in data:
            n = x["iteration"]
            want_hits = (9+int(mode in ("same","side") and n>0)
                         if mode in ("S2","same","side") else 0)
            assert x["hits"]==want_hits and x["qkv_calls"]==312, (mode, rank, n, x["qkv_calls"])
            assert len(x["reduce_scatter"])==37
            if n:
                assert all(y["stage"]==["backward",7] and y["async_op"]
                           for y in x["reduce_scatter"])
        rank_rows.append(dict(rank=rank, allocated=max(x["allocated"] for x in data),
            reserved=max(x["reserved"] for x in data), hits=sum(x["hits"] for x in data),
            released=sum(x["released"] for x in data), loss=[x["losses"] for x in data],
            target_consumptions=sum(x.get("target_consumed",False) for x in data),
            max_cached=max(x["max_cached"] for x in data)))
    key = mode if rep=="final" else f"{mode}_{rep}"
    reports["perf"][key] = dict(ranks=rank_rows,steady_ms=stats(times[5:]),all_times_ms=times,
        max_allocated=max(x["allocated"] for x in rank_rows),max_reserved=max(x["reserved"] for x in rank_rows))
    if mode!="S0" and "S0" in reports["perf"]:
        ref=reports["perf"]["S0"]
        assert [x["loss"] for x in ref["ranks"]]==[x["loss"] for x in rank_rows]
paired = []
for suffix in ("","_r2","_r3"):
    same = reports["perf"].get("same"+suffix)
    side = reports["perf"].get("side"+suffix)
    if same and side:
        base = same["steady_ms"]["median"]
        delta = side["steady_ms"]["median"]-base
        paired.append(dict(run=suffix or "final", delta_ms=delta, delta_pct=delta/base*100,
                           allocated_bytes=side["max_allocated"]-same["max_allocated"],
                           reserved_bytes=side["max_reserved"]-same["max_reserved"]))
reports["paired_perf"] = paired
(root/"poc_summary.json").write_text(json.dumps(reports,indent=2))
for k,r in reports["timeline"].items():
    print(k, {col:stats([x[col] for x in r]) for col in 
              ("overlap_ms","first_bucket_overlap_ms","replay_ms","first_rs_ms","resident_delta",
               "target_lifetime_cpu_ms","replay_to_consume_event_ms")})
for k,r in reports["perf"].items():
    print(k,r["steady_ms"],r["max_allocated"],r["max_reserved"])
