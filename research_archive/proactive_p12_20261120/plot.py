"""Plot measured final Backward/DP/optimizer intervals around the terminal gate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(2,2,figsize=(11,6),layout="constrained")
for ax, mode in zip(axes.flat,("C0","Cnoop","C1","Clate")):
    d = root/f"p12_4b_{mode}_trace_conn1"
    r = json.loads((root/(d.name+"_exposure.json")).read_text())["ranks"][0]
    bwd = [e for e in json.loads((d/"nvtx_attribution.json").read_text())["0"] if
           e["kind"]=="backward" and e["stage"]==["backward",7]][0]
    qkv = [e for e in json.loads((d/"nvtx_attribution.json").read_text())["0"]
           if e["kind"]=="qkv" and e["layer"]==1 and
           e["stage"] in (["backward",7],["prefetch",7])][0]
    comm = json.loads((d/"comm.json").read_text())
    rs = [k for k in comm if k["rank"]==0 and k["kind"]=="grad_rs"]
    zero = r["final_backward_with_tp_end_ns"]
    to_ms = lambda t:(t-zero)/1e6
    def bars(events,y,color):
        if events:
            ax.broken_barh([(to_ms(k["start"]),(k["end"]-k["start"])/1e6) for k in events
                            if to_ms(k["end"])>-9],(y,.45),facecolors=color)
    bars(rs,.05,"#d08e36")
    bars([k for k in bwd["kernels"] if "nccl" not in k["name"].lower()],.55,"#739bbd")
    bars(qkv["kernels"],1.05,"#25765e")
    ax.axvspan(0,to_ms(r["last_rs_end_ns"]),alpha=.13,color="#d08e36")
    ax.axvline(0,ls=":",lw=1,color="#334155")
    ax.axvline(to_ms(r["first_opt_gpu_ns"]),ls="--",lw=1,color="#945987",label="First optimizer kernel")
    ax.set_yticks([.25,.75,1.25],["DP RS","Backward math","Target Replay"])
    ax.set_xlim(-9,max(6,to_ms(r["first_opt_gpu_ns"])+1))
    ax.set_ylim(0,1.55)
    ax.set_title(mode,loc="left",fontsize=10)
    ax.set_xlabel("GPU time relative to last Backward compute/TP end (ms)")
    ax.grid(axis="x",alpha=.2)
fig.suptitle("Qwen3-4B · rank 0, step 3, final micro-batch · shaded DP terminal window",fontsize=12)
fig.savefig(root/"critical_path_timeline.png",dpi=180)
