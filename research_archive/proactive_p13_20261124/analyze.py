"""Attribute GPU kernels via NVTX -> CUDA API -> CUPTI correlation IDs."""
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

case = Path(sys.argv[1])
cx = sqlite3.connect(case/"trace.sqlite")
names = dict(cx.execute("select id,value from StringIds"))
kernels = defaultdict(list)
for start, end, pid, corr, stream, name in cx.execute(
    "select start,end,globalPid,correlationId,streamId,shortName from CUPTI_ACTIVITY_KIND_KERNEL"
):
    kernels[pid>>24, corr].append(dict(start=start, end=end, stream=stream, name=names[name],
                                      pid=pid>>24, correlation=corr))
launch = defaultdict(list)
for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    if not cx.execute("select 1 from sqlite_master where name=?", (table,)).fetchone():
        continue
    for start, tid, corr in cx.execute(f"select start,globalTid,correlationId from {table}"):
        if (tid>>24, corr) in kernels:
            launch[tid].append((start,corr))
by_process = defaultdict(list)
for tid, v in launch.items():
    v.sort()
    by_process[tid>>24].extend(v)
for v in by_process.values():
    v.sort()

def union(ranges):
    out = []
    for start,end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1],end)
        elif start < end:
            out.append([start,end])
    return out

def intervals(items):
    return union((k["start"],k["end"]) for k in items)

def duration(ranges):
    return sum(b-a for a,b in ranges)/1e6

def intersect(x,y):
    x,y = union(x),union(y)
    intervals = []
    i = j = 0
    while i < len(x) and j < len(y):
        a,b = x[i]
        c,d = y[j]
        if max(a,c) < min(b,d):
            intervals.append([max(a,c),min(b,d)])
        i += b <= d
        j += d <= b
    return intervals

def kernel_id(k):
    return tuple(k[x] for x in ("pid","stream","start","end","correlation"))

def is_nccl(k):
    return "nccl" in k["name"].lower()

def operation(k):
    name = k["name"].lower()
    return next((x for x in ("allreduce","allgather","reducescatter","broadcast")
                 if x in name.replace("_","")), "other")

rows = defaultdict(list)
for a,b,t,tid in cx.execute(
    "select start,end,text,globalTid from NVTX_EVENTS where text like 'P10|%' and end is not null"
):
    meta = json.loads(t[4:])
    # Autograd workers submit kernels inside the parent backward range on
    # other host threads. Leaf NVTX scopes use exact TID attribution.
    candidates = (by_process[tid>>24] if meta["kind"] in ("step","forward","backward")
                  else launch[tid])
    lo, hi = [bisect.bisect_left(candidates,(v,-1)) for v in (a,b)]
    matched = {kernel_id(k): k for _,corr in candidates[lo:hi] for k in kernels[tid>>24,corr]}
    rows[meta["rank"]].append(dict(meta,cpu_start=a,cpu_end=b,kernels=list(matched.values())))
assert sorted(rows) == list(range(8)), sorted(rows)
summary = []
tasks = []
comm = []
for rank, marks in sorted(rows.items()):
    step, = [x for x in marks if x["kind"] == "step"]
    mb = [x for x in marks if x["kind"] in ("forward","backward")]
    qkv = [x for x in marks if x["kind"] == "qkv"]
    dp = [x for x in marks if x["kind"] == "grad_sync"]
    prefetch = [x for x in marks if x["kind"] == "prefetch"]
    def bucket(events):
        return {kernel_id(k):k for e in events for k in e["kernels"] if is_nccl(k)}
    dp_k, handshake_k = bucket(dp), bucket(prefetch)
    compute = [k for k in step["kernels"] if not is_nccl(k)]
    collectives = bucket([step])
    seen = bucket(mb)
    registered = set(dp_k)|set(handshake_k)
    model_k = {ident:k for ident,k in seen.items() if ident not in registered}
    unknown = [k for ident,k in collectives.items() if ident not in (registered|set(seen))]
    replay_k = [k for x in qkv if x["stage"][0] == "prefetch" for k in x["kernels"] if not is_nccl(k)]
    replay_nccl = bucket([x for x in qkv if x["stage"][0] == "prefetch"])
    assert not replay_nccl, replay_nccl
    assert all(operation(k)=="reducescatter" for k in dp_k.values())
    assert dp_k and all(x["stage"]==["backward",7] for x in dp if bucket([x]))
    assert len(mb) == 16
    qkv_orig = {(x["stage"][1], x["layer"]):x for x in qkv if x["stage"][0]=="forward"}
    assert len(qkv_orig) == len([x for x in qkv if x["stage"][0]=="forward"])
    dp_ranges, model_ranges, replay_ranges = [intervals(ks) for ks in (dp_k.values(),model_k.values(),replay_k)]
    launch_windows = []
    for i, event in enumerate(sorted(dp,key=lambda e:e["cpu_start"])):
        for k in bucket([event]).values():
            launch_windows.append((i,max(k["start"],event["cpu_end"]),k["end"]))
    tp_launch_windows = []
    for event in marks:
        if event["kind"] == "all_reduce_api":
            for k in bucket([event]).values():
                if kernel_id(k) in model_k:
                    tp_launch_windows.append((max(k["start"],event["cpu_end"]),k["end"]))
    for e in qkv:
        stage, micro = e["stage"]
        if stage == "forward":
            continue
        orig = qkv_orig[micro,e["layer"]]
        if not e["kernels"] or not orig["kernels"]:
            raise RuntimeError((rank,e["stage"],e["layer"]))
        ready_upper = min(k["start"] for k in orig["kernels"])
        start = min(k["start"] for k in e["kernels"])
        end = max(k["end"] for k in e["kernels"])
        back = next(x for x in mb if x["stage"]==["backward",micro])
        # On-demand replay starts at the actual consumption site. For proactive
        # replay, use first gradient-side kernel after the original QKV site only
        # when known. This conservative bound is the start of the entire backward.
        deadline = start if stage=="backward" else min(k["start"] for k in back["kernels"])
        allowed = [(max(ready_upper,a),min(deadline,b)) for _,a,b in launch_windows]
        potential = intersect(union(allowed),dp_ranges)
        eligible = [dict(bucket=i,left_ms=(min(deadline,b)-max(ready_upper,a))/1e6)
                    for i,a,b in launch_windows if max(ready_upper,a)<min(deadline,b)]
        tasks.append(dict(rank=rank,stage=stage,mb=micro,layer=e["layer"],ready_upper_ns=ready_upper,
                          deadline_ns=deadline,gpu_start_ns=start,gpu_end_ns=end,
                          replay_gpu_ms=duration(intervals(e["kernels"])),dp_after_host_return_ms=duration(potential),
                          dp_candidates=eligible,
                          compute_streams=sorted({k["stream"] for k in e["kernels"]}),
                          dp_actual_ms=duration(intersect(intervals(e["kernels"]),dp_ranges)),
                          tp_actual_ms=duration(intersect(intervals(e["kernels"]),model_ranges)),
                          tp_after_host_return_ms=duration(intersect([(ready_upper,deadline)],union(tp_launch_windows))),
                          tp_longest_after_host_return_ms=max((b-a for a,b in intersect(
                              [(ready_upper,deadline)],union(tp_launch_windows))),default=0)/1e6,
                          tp_window_upper_ms=duration(intersect([(ready_upper,deadline)],model_ranges))))
    dp_stage = {ident: event["stage"] for event in dp for ident in bucket([event])}
    mb_stage = {ident: event["stage"] for event in mb for ident in bucket([event])}
    for kind, items in (("grad_rs",dp_k),("tp_model",model_k),("tp_handshake",handshake_k)):
        for k in items.values():
            dispatch = (dp_stage if kind=="grad_rs" else mb_stage).get(kernel_id(k))
            comm.append(dict(rank=rank,kind=kind,operation=operation(k),stage=dispatch,**k))
    summary.append(dict(rank=rank,dp_count=len(dp_k),tp_count=len(model_k),handshake_count=len(handshake_k),
                        dp_ms=duration(dp_ranges),tp_ms=duration(model_ranges),
                        dp_native_compute_intersect_ms=duration(intersect(intervals(compute),dp_ranges)),
                        dp_replay_intersect_ms=duration(intersect(replay_ranges,dp_ranges)),
                        tp_replay_intersect_ms=duration(intersect(replay_ranges,model_ranges)),
                        dp_ms_by_kernel=[(k["end"]-k["start"])/1e6 for k in dp_k.values()],
                        tp_ops={op:dict(n=len(ks),gpu_ms=duration(intervals(ks))) for op in sorted({operation(k) for k in model_k.values()})
                                for ks in [[k for k in model_k.values() if operation(k)==op]]},
                        prefetch_to_first_dp_ms=(dp_ranges[0][0]-replay_ranges[-1][1])/1e6 if replay_ranges else None,
                        replay_ms=duration(replay_ranges),
                        forward_backward_tp_api_tagged_count=sum(
                            len([k for k in bucket([x]).values() if kernel_id(k) in model_k])
                            for x in marks if x["kind"]=="all_reduce_api"),
                        unknown_nccl=[dict(name=k["name"],operation=operation(k)) for k in unknown]))
for name,content in (("trace_analysis",summary),("tasks",tasks),("comm",comm),("nvtx_attribution",rows)):
    (case/(name+".json")).write_text(json.dumps(content,indent=2))
print(case, len(summary), "ranks",len(tasks),"QKV replays")
