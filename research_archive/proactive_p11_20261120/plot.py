"""Render a CUPTI-kernel zoom around the chosen RS and target QKV launches."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
summary = json.loads((root/"summary.json").read_text())["timeline"]
rank = max(range(8),key=lambda i:sum(next(r["overlap_ms"] for r in rows if r["rank"]==i)
                                    for mode,rows in summary.items() if mode.startswith("C1")))
fig, axes = plt.subplots(2,1,figsize=(9,4),layout="constrained")
for ax,mode,label,ix in zip(axes, ("C1early","C1late"),("First RS","36th RS"),(0,35)):
    p = root/f"p11_4b_{mode}_trace_conn1"
    marks = json.loads((p/"nvtx_attribution.json").read_text())[str(rank)]
    bucket = sorted([e for e in marks if e["kind"]=="grad_sync"],key=lambda e:e["cpu_start"])[ix]
    rs, = [k for k in bucket["kernels"] if "nccl" in k["name"].lower()]
    qkv, = [e for e in marks if e["kind"]=="qkv" and e["stage"]==["prefetch",7] and e["layer"]==1]
    replay = qkv["kernels"]
    zero = min(k["start"] for k in replay)
    def rel(ns):
        return (ns-zero)/1e6
    for k in replay:
        a,b = k["start"],k["end"]
        ax.broken_barh([(rel(a),(b-a)/1e6)],(1.03,.46),facecolors="#377bb5")
        a,b = max(a,rs["start"]),min(b,rs["end"])
        if a<b:
            ax.broken_barh([(rel(a),(b-a)/1e6)],(.03,.46),facecolors="#248a6f")
    ax.broken_barh([(rel(rs["start"]),(rs["end"]-rs["start"])/1e6)],(.53,.46),facecolors="#c27e24")
    ax.set_yticks([.25,.75,1.25],["Intersection","DP Reduce-Scatter","QKV Replay"])
    ax.set_xlim(-.18,max(rel(k["end"]) for k in replay)+.25)
    ax.set_ylim(0,1.55)
    ax.grid(axis="x",alpha=.25)
    ax.set_title(label,loc="left",fontsize=10)
    ax.set_xlabel("GPU time relative to the first Replay kernel (ms)")
fig.suptitle(f"Native PP1 · Qwen3-4B · rank {rank}, step 3, micro-batch 8",fontsize=12)
fig.savefig(root/"native_pp1_kernel_overlap.png",dpi=190)
