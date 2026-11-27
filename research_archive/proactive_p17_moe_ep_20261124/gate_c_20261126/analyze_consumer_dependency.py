#!/usr/bin/env python3
"""Relate a target EP Combine kernel to its downstream output consumer."""
import json
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
db.execute("create index if not exists p17_nvtx_target on NVTX_EVENTS(text, start)")
db.execute("create index if not exists p17_runtime_thread on CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start)")
db.execute("create index if not exists p17_kernel_corr on CUPTI_ACTIVITY_KIND_KERNEL(globalPid, correlationId)")
db.execute("create index if not exists p17_kernel_time on CUPTI_ACTIVITY_KIND_KERNEL(start, end)")
rows = []
for step in (20, 21, 22):
    combines = list(db.execute("select start,end,globalTid from NVTX_EVENTS where text=?", (f"P17_TARGET_COMBINE_{step}",)))
    consumers = list(db.execute("select start,end,globalTid from NVTX_EVENTS where text=?", (f"P17_TARGET_CONSUMER_{step}",)))
    for a, b, tid in combines:
        pid = tid & ~16777215
        consumer = min((r for r in consumers if (r[2] & ~16777215) == pid and r[0] >= a), key=lambda x: x[0], default=None)
        if consumer is None:
            continue
        q = """select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
               join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=? and k.correlationId=r.correlationId
               join StringIds s on s.id=k.shortName where r.globalTid=? and r.start between ? and ? order by k.start"""
        combine_kernels = list(db.execute(q, (pid, tid, a, b)))
        nccl = [x for x in combine_kernels if "nccl" in x[3].lower()]
        if not nccl:
            continue
        comm_start = min(x[0] for x in nccl)
        comm_end = max(x[1] for x in nccl)
        ca, cb, ctid = consumer
        consumer_kernels = list(db.execute(q, (pid, ctid, ca, cb)))
        if not consumer_kernels:
            continue
        first_consumer = min(consumer_kernels, key=lambda x: x[0])
        between = list(db.execute("select start,end,streamId,shortName from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=? and start>=? and end<=? order by start", (pid, comm_end, first_consumer[0])))
        replay = list(db.execute("""select k.start,k.end,k.streamId,s.value from NVTX_EVENTS n
                join CUPTI_ACTIVITY_KIND_RUNTIME r on r.globalTid=n.globalTid and r.start between n.start and n.end
                join CUPTI_ACTIVITY_KIND_KERNEL k on k.globalPid=? and k.correlationId=r.correlationId
                join StringIds s on s.id=k.shortName where (n.globalTid & -16777216)=? and n.text=?""", (pid, pid, f"proactive_recompute.prefetch.{step}")))
        rows.append({
            "pid": pid, "step": step,
            "combine_streams": sorted({x[2] for x in nccl}),
            "combine_kernel": nccl[0][3],
            "combine_kernel_us": (comm_end - comm_start) / 1000,
            "consumer_stream": first_consumer[2],
            "first_consumer_kernel": first_consumer[3],
            "combine_end_to_consumer_kernel_us": (first_consumer[0] - comm_end) / 1000,
            "kernels_between_comm_end_and_consumer": len(between),
            "replay_kernels": [{"start": x[0], "end": x[1], "stream": x[2], "name": x[3]} for x in replay],
            "consumer_nvtx_start_minus_combine_kernel_end_us": (ca - comm_end) / 1000,
            "combine_nvtx_cpu_end_to_kernel_end_us": (comm_end - b) / 1000,
        })
gaps = [r["combine_end_to_consumer_kernel_us"] for r in rows]
summary = {
    "samples": len(rows),
    "combine_end_to_consumer_kernel_us": {
        "mean": statistics.mean(gaps), "median": statistics.median(gaps),
        "min": min(gaps), "max": max(gaps),
    } if gaps else None,
    "samples_with_kernels_between": sum(r["kernels_between_comm_end_and_consumer"] > 0 for r in rows),
    "note": "A gap with no CUDA kernel is not by itself proof of whole-GPU idleness or end-to-end critical-path exposure.",
}
json.dump({"summary": summary, "samples": rows}, sys.stdout, indent=2)
print()
