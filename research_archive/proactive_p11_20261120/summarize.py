"""Audit native schedule and collect replay/RS GPU intersections and timing."""
from pathlib import Path
import json
import re
import statistics

root = Path(__file__).resolve().parent
modes = {"S0":"C0","C1early":"C1early","C1late":"C1late","same":"C2"}

def read(case, name):
    return json.loads((case/name).read_text())

def stats(values):
    return dict(n=len(values), min=min(values), median=statistics.median(values), max=max(values))

def total_intersection(replays, rs):
    return sum(max(0,min(k["end"],rs["end"])-max(k["start"],rs["start"])) for k in replays)/1e6

native = [[phase, i] for i in range(8) for phase in ("forward","backward")]
windowed = [pair for start in (0,2,4)
            for pair in (["forward",start],["forward",start+1],["backward",start],["backward",start+1])]
windowed.extend(native[12:])
results = {"timeline":{},"runs":{}}
for mode in ("C1early","C1late","same"):
    case = root/f"p11_4b_{mode}_trace_conn1"
    if not (case/"tasks.json").exists():
        continue
    marks = read(case,"nvtx_attribution.json")
    baseline_marks = read(root/"p11_4b_S0_trace_conn1","nvtx_attribution.json")
    tasks = read(case,"tasks.json")
    data = read(case,"trace_analysis.json")
    rows = []
    ix = 35 if mode=="C1late" else 0
    for rank in range(8):
        r = read(case,f"rank{rank}.json")["steps"][-1]
        assert r["schedule"] == (windowed if mode=="same" else native)
        cp, = [v for v in r["checkpoints"] if v["metadata"]["microbatch"]==7
               and v["metadata"]["layer_number"]==1]
        ev = {v["event"]:v for v in cp["events"]}
        assert cp["hit"] and cp["final_state"]=="consumed" and "fallback" not in ev
        assert len([x for x in cp["events"] if x["event"]=="cache_consumed"])==1
        assert not any(x["rank"]==rank and x["stage"]=="backward" and
                       x["mb"]==7 and x["layer"]==1 for x in tasks)
        orig, replay = [x for stage in ("forward","prefetch") for x in marks[str(rank)]
                        if x["kind"]=="qkv" and x["stage"]==[stage,7] and x["layer"]==1]
        mark, = [x for x in marks[str(rank)] if x["kind"]=="prefetch" and
                 x["checkpoint_id"]==cp["id"]]
        dispatch = sorted((x for x in marks[str(rank)] if x["kind"]=="grad_sync"),
                          key=lambda e:e["cpu_start"])[ix]
        assert dispatch["cpu_end"] <= mark["cpu_start"]
        rs, = [v for v in dispatch["kernels"] if "nccl" in v["name"].lower()]
        time = {name:ms for name,_,ms in r["timeline"]}
        ready = time[f"replay:{cp['id']}:7:prefetch_complete"]
        consume = time[f"consume:{cp['id']}"]
        release = time[f"release:{cp['id']}"]
        assert ready < consume < release
        baseline_bwd, = [e for e in baseline_marks[str(rank)] if e["kind"]=="backward"
                         and e["stage"]==["backward",7]]
        native_streams = {k["stream"] for k in baseline_bwd["kernels"]
                          if "nccl" not in k["name"].lower()}
        replay_streams = {k["stream"] for k in replay["kernels"]}
        assert replay_streams <= native_streams and rs["stream"] not in replay_streams
        memory = ev["released"]["memory"]
        rows.append(dict(rank=rank, bucket=ix, group=dispatch["group"],
                         replay_streams=sorted(replay_streams), rs_stream=rs["stream"],
                         overlap_ms=total_intersection(replay["kernels"],rs),
                         all_dp_overlap_ms=next(x["dp_actual_ms"] for x in tasks if x["rank"]==rank and
                                                x["stage"]=="prefetch" and x["mb"]==7 and x["layer"]==1),
                         replay_ms=sum(k["end"]-k["start"] for k in replay["kernels"])/1e6,
                         rs_ms=(rs["end"]-rs["start"])/1e6, rs_gpu_end=rs["end"],
                         first_replay_gpu_start=min(k["start"] for k in replay["kernels"]),
                         rs_submit_cpu_end=dispatch["cpu_end"],
                         lifetime_gpu_ms=consume-ready,
                         lifetime_cpu_ms=ev["released"]["cache_lifetime_seconds"]*1000,
                         allocated_delta=memory["allocated_after"]-memory["allocated_before"],
                         native_dp_compute_intersection_ms=data[rank]["dp_native_compute_intersect_ms"]))
    results["timeline"][mode] = rows

losses = {}
orders = {}
for mode in modes:
    for run in range(1,4):
        case = root/f"p11_4b_{mode}_perf_conn1_r{run}"
        if not all((case/f"rank{i}.json").exists() for i in range(8)):
            continue
        lines = (case/"train.log").read_text()
        times = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)",lines)]
        assert len(times)==12 and not re.search(r"skipped iterations:\s+[1-9]|nan iterations:\s+[1-9]",lines)
        rows = []
        for rank in range(8):
            steps = read(case,f"rank{rank}.json")["steps"]
            assert len(steps)==12
            for r in steps:
                iteration = r["iteration"]
                key = (rank,iteration)
                assert r["losses"] == losses.setdefault(key,r["losses"]), (mode,run,key)
                expected = (int(iteration>0) if mode.startswith("C1") else
                            9+int(iteration>0) if mode=="same" else 0)
                assert r["hits"]==expected and r["qkv_calls"]==312, (mode,run,key,r["hits"])
                assert r["released"] == (1 if mode.startswith("C1") else 24 if mode=="same" else 0)
                assert r["schedule"] == (windowed if mode=="same" else native), (mode,run,key)
                assert len(r["reduce_scatter"])==37
                if iteration:
                    assert all(x["stage"]==["backward",7] and x["async_op"] for x in r["reduce_scatter"])
                    order = [x["group"] for x in r["sync"]]
                    assert len(order)==37 and order==orders.setdefault(key,order)
                if mode.startswith("C1") and iteration:
                    cp, = r["checkpoints"]
                    names = [e["event"] for e in cp["events"]]
                    assert all(names.count(e)==1 for e in ("forward_complete","prefetch_complete",
                                                           "cache_consumed","released"))
                    assert "fallback" not in names and cp["final_state"]=="consumed"
            rows.append(dict(rank=rank, max_allocated=max(x["allocated"] for x in steps),
                             max_reserved=max(x["reserved"] for x in steps),
                             hits=sum(x["hits"] for x in steps),
                             max_cached=max(x["max_cached"] for x in steps)))
        results["runs"].setdefault(mode,{})[run] = dict(ranks=rows, all_steps_ms=times,
            steady_ms=stats(times[5:]),max_allocated=max(x["max_allocated"] for x in rows),
            max_reserved=max(x["max_reserved"] for x in rows))

results["paired"] = {}
for mode in ("C1early","C1late","same"):
    rows = results["runs"].get(mode,{})
    base = results["runs"].get("S0",{})
    common = sorted(set(rows)&set(base))
    if common:
        results["paired"][mode] = {r:dict(
            delta_ms=rows[r]["steady_ms"]["median"]-base[r]["steady_ms"]["median"],
            allocated_delta=rows[r]["max_allocated"]-base[r]["max_allocated"],
            reserved_delta=rows[r]["max_reserved"]-base[r]["max_reserved"]) for r in common}
(root/"summary.json").write_text(json.dumps(results,indent=2))
for mode, rows in results["timeline"].items():
    print(mode,{k:stats([r[k] for r in rows]) for k in ("overlap_ms","all_dp_overlap_ms",
          "replay_ms","rs_ms","lifetime_gpu_ms","allocated_delta")})
for mode, runs in results["runs"].items():
    print(modes[mode],[(i,r["steady_ms"],r["max_allocated"],r["max_reserved"]) for i,r in runs.items()])
print(results["paired"])
