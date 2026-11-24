"""Plot the intersection of individual CUPTI kernel intervals on rank 0."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 1, figsize=(9, 4), layout="constrained")
for ax, mode, label in zip(axes, ("same","side"), ("S2-DP · main compute stream", "S3 · separate replay stream")):
    path = base/f"poc_4b_{mode}_trace_conn1_fenced"
    events = json.loads((path/"nvtx_attribution.json").read_text())["0"]
    qkv, = [r for r in events if r["kind"] == "qkv" and r["stage"] == ["prefetch",7] and r["layer"] == 1]
    comm = json.loads((path/"comm.json").read_text())
    rs = min((r for r in comm if r["rank"] == 0 and r["kind"] == "grad_rs"), key=lambda r:r["start"])
    replay = sorted(qkv["kernels"], key=lambda r:r["start"])
    zero = replay[0]["start"]
    rel = lambda v: (v-zero)/1e6
    for kernel in replay:
        a,b = kernel["start"], kernel["end"]
        ax.broken_barh([(rel(a), (b-a)/1e6)], (1.03,.46), facecolors="#3b82c4")
        overlap = max(0,min(b,rs["end"])-max(a,rs["start"]))
        if overlap:
            ax.broken_barh([(rel(max(a,rs["start"])), overlap/1e6)], (.03,.46), facecolors="#218a73")
    ax.broken_barh([(rel(rs["start"]),(rs["end"]-rs["start"])/1e6)], (.53,.46), facecolors="#d48b2c")
    ax.set_xlim(-.15, max(rel(k["end"]) for k in replay)+.16)
    ax.set_ylim(0,1.56)
    ax.set_yticks([.25,.75,1.25],
                  ["Intersection",f"DP RS · stream {rs['stream']}",f"Replay · stream {replay[0]['stream']}"])
    ax.set_title(label, loc="left", fontsize=10)
    ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("GPU time relative to first Replay kernel (ms)")
    ax.spines[["top","right","left"]].set_visible(False)
fig.suptitle("Qwen3-4B · rank 0, step 3, last micro-batch, QKV layer 1", fontsize=12)
fig.savefig(base/"kernel_overlap_rank0.png", dpi=190)
