"""Validate all repeated runs and compare matched elapsed time and memory peaks."""
from pathlib import Path
import json
import re
import statistics as st

root = Path(__file__).resolve().parent
modes = ("C0","Cnoop","C1","Clate")
native = [[stage, mb] for mb in range(8) for stage in ("forward","backward")]
all_rows, orders, losses = {}, {}, {}
for mode in modes:
    all_rows[mode]={}
    for run in range(1,6):
        case=root/f"p12_4b_{mode}_perf_conn1_r{run}"
        if not all((case/f"rank{rank}.json").exists() for rank in range(8)):
            continue
        txt=(case/"train.log").read_text()
        elapsed=[float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)",txt)]
        assert len(elapsed)==10 and not re.search(r"skipped iterations:\s+[1-9]|nan iterations:\s+[1-9]",txt), case
        ranks=[]
        for rank in range(8):
            steps=json.loads((case/f"rank{rank}.json").read_text())["steps"]
            assert len(steps)==10
            for step in steps:
                iteration=step["iteration"]
                key=(rank,iteration)
                assert losses.setdefault(key,step["losses"])==step["losses"]
                assert step["schedule"]==native and step["qkv_calls"]==312
                assert len(step["reduce_scatter"])==37
                assert step["hits"]==int(iteration>0 and mode in ("C1","Clate"))
                assert step["trigger_count"]==int(iteration>0 and mode!="C0")
                assert step["released"]==int(mode!="C0")
                if iteration:
                    seq=[r["group"] for r in step["sync"]]
                    assert seq==orders.setdefault(key,seq)
                    assert all(r["stage"]==["backward",7] and r["async_op"]
                               for r in step["reduce_scatter"])
                if mode!="C0":
                    cp,=step["checkpoints"]
                    events=[e["event"] for e in cp["events"]]
                    assert cp["final_state"]=="consumed"
                    assert events.count("released")==1
                    if step["hits"]:
                        assert events.count("cache_consumed")==1 and "fallback" not in events
                    else:
                        assert events.count("fallback")==1 and "cache_consumed" not in events
            ranks.append(dict(rank=rank, allocated=max(x["allocated"] for x in steps),
                              reserved=max(x["reserved"] for x in steps),
                              max_cached=max(x["max_cached"] for x in steps),
                              checkpoints=sum(x["released"] for x in steps),
                              hits=sum(x["hits"] for x in steps),
                              triggers=sum(x["trigger_count"] for x in steps)))
        all_rows[mode][run]=dict(median_ms=st.median(elapsed[5:]),steps_ms=elapsed,ranks=ranks,
              allocated=max(x["allocated"] for x in ranks),reserved=max(x["reserved"] for x in ranks))
pairs={}
for target,reference in (("Cnoop","C0"),("C1","C0"),("Clate","C0"),("C1","Cnoop"),("Clate","Cnoop"),
                         ("Clate","C1")):
    d=[]
    for run in sorted(set(all_rows[target])&set(all_rows[reference])):
        a,b=all_rows[target][run],all_rows[reference][run]
        d.append(dict(run=run,delta_ms=a["median_ms"]-b["median_ms"],
                      allocated_delta=a["allocated"]-b["allocated"],
                      reserved_delta=a["reserved"]-b["reserved"]))
    pairs[target+"-"+reference]=d
(root/"performance_summary.json").write_text(json.dumps(dict(runs=all_rows,paired=pairs),indent=2))
for mode,data in all_rows.items():
    print(mode,[(i,r["median_ms"]) for i,r in data.items()])
for mode,data in pairs.items():
    if data:
        d=[x["delta_ms"] for x in data]
        print(mode,len(d),[round(x,2) for x in d],round(st.median(d),2))
