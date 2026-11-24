"""Audit CUDA wait API submissions inside the DP finalize ranges."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

root = Path(__file__).resolve().parent
report = {}
for mode in ("C0","Cnoop","C1","Clate"):
    cx=sqlite3.connect(root/f"p12_4b_{mode}_trace_conn1"/"trace.sqlite")
    scopes=defaultdict(list)
    for a,b,t,tid in cx.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'P10|%' and end is not null"):
        v=json.loads(t[4:])
        if v["kind"]=="grad_finish":
            scopes[tid].append((a,b,v["rank"],v["group"]))
    waits=defaultdict(list)
    for a,b,tid,n in cx.execute("""select r.start,r.end,r.globalTid,s.value
        from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
        where s.value like '%StreamWaitEvent%' or s.value like '%EventSynchronize%'"""):
        for x,y,rank,group in scopes[tid]:
            if x<=a<y:
                waits[rank].append(dict(group=group,api=n,start=a,end=b))
                break
    assert len(waits)==8
    report[mode]={rank:rows for rank,rows in sorted(waits.items())}
    print(mode,[len(x) for x in waits.values()])
(root/"cuda_grad_finish_wait_api.json").write_text(json.dumps(report,indent=2))
