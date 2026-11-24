"""Use the median of independent run medians and report the run range."""
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
modes = {mode: [] for mode in ("S0","S1","S2")}
for r in range(1,4):
    for mode in modes:
        case = root/f"perf_mb8_r{r}"/mode
        times = [float(t) for t in re.findall(
            r"elapsed time per iteration \(ms\): ([\d.]+)", (case/"train.log").read_text())]
        assert len(times)==30
        ranks = [json.loads((case/f"rank{i}.json").read_text())["steps"] for i in range(8)]
        assert all(len(x)==30 and all(y["qkv_calls"]==32 for y in x) for x in ranks)
        row = dict(run=r,median_ms=statistics.median(times[5:]),mean_ms=statistics.mean(times[5:]),
                   min_ms=min(times[5:]),max_ms=max(times[5:]),times_ms=times,
                   by_rank=[dict(
                       allocated=max(y["allocated"] for y in x), reserved=max(y["reserved"] for y in x),
                       forward=sum(v for y in x for k,v in y["calls"].items() if "'forward'" in k),
                       prefetch=sum(v for y in x for k,v in y["calls"].items() if "'prefetch'" in k),
                       backward_replay=sum(v for y in x for k,v in y["calls"].items() if "'backward'" in k),
                       hits=sum(y["hits"] for y in x), released=sum(y["released"] for y in x),
                       max_cached=max(y["max_cached"] for y in x)) for x in ranks])
        for key in ("allocated","reserved"):
            row[key+"_max"]=max(y[key] for y in row["by_rank"])
        modes[mode].append(row)
def stats(values):
    return dict(median=statistics.median(values), min=min(values),max=max(values))
summary = {mode:{key:stats([row[key] for row in rows])
                 for key in ("median_ms","mean_ms","allocated_max","reserved_max")}
           for mode,rows in modes.items()}
pairs={}
for a,b in (("S1","S0"),("S2","S1"),("S2","S0")):
    pairs[a+"-"+b]={key:stats([x[key]-y[key] for x,y in zip(modes[a],modes[b])])
                    for key in ("median_ms","allocated_max","reserved_max")}
    pairs[a+"-"+b]["relative_time_pct"]=stats([(x["median_ms"]/y["median_ms"]-1)*100
                                             for x,y in zip(modes[a],modes[b])])
(root/"perf_summary.json").write_text(json.dumps(dict(runs=modes,summary=summary,paired=pairs),indent=2))
print(json.dumps(dict(summary=summary,paired=pairs),indent=2))
