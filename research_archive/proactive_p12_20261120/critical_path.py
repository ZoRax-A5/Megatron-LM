"""Measure trace exposure bounds, not a counterfactual from kernel overlap."""
import json
from pathlib import Path
import statistics
import sys

def merge(xs):
    out = []
    for a,b in sorted(xs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1],b)
        elif a<b:
            out.append([a,b])
    return out

def overlap(a,b):
    a,b = merge(a),merge(b)
    out = []
    i=j=0
    while i<len(a) and j<len(b):
        s,e=a[i]
        u,v=b[j]
        if max(s,u)<min(e,v):
            out.append((max(s,u),min(e,v)))
        i += e<=v
        j += v<=e
    return out

def ns(x):
    return sum(b-a for a,b in merge(x))

def iv(ks):
    return [(k["start"],k["end"]) for k in ks]

def nccl(k):
    return "nccl" in k["name"].lower()

def uid(k):
    return tuple(k[n] for n in ("pid","correlation","start","end","stream"))

def stat(xs):
    xs = list(xs)
    return dict(min=min(xs),median=statistics.median(xs),max=max(xs))

case = Path(sys.argv[1])
marks = json.loads((case/"nvtx_attribution.json").read_text())
comm = json.loads((case/"comm.json").read_text())
ranks = []
for rank in range(8):
    rows = marks[str(rank)]
    bwd, = [x for x in rows if x["kind"]=="backward" and x["stage"]==["backward",7]]
    step, = [x for x in rows if x["kind"]=="step"]
    sectors = [x for x in comm if x["rank"]==rank and x["kind"]=="grad_rs"]
    model = [x for x in comm if x["rank"]==rank and x["kind"]=="tp_model"]
    assert len(sectors)==37
    dp = sorted(sectors,key=lambda k:k["start"])
    math = [k for k in bwd["kernels"] if not nccl(k)]
    tp = [x for x in model if x["stage"]==["backward",7]]
    end_math = max(k["end"] for k in math)
    end_deps = max(end_math,max(k["end"] for k in tp))
    rs_end = max(k["end"] for k in dp)
    qkv, = [x for x in rows if x["kind"]=="qkv" and x["layer"]==1 and
            x["stage"] in (["backward",7],["prefetch",7])]
    qstart = min(k["start"] for k in qkv["kernels"])
    qend = max(k["end"] for k in qkv["kernels"])
    waits = [x for x in rows if x["kind"]=="grad_finish"]
    optimizer = [x for x in rows if x["kind"]=="optimizer"]
    nopt = [k for x in optimizer for k in x["kernels"]]
    opt_start = min((k["start"] for k in nopt),default=None)
    opt_math = min((k["start"] for k in nopt if not nccl(k)),default=None)
    grad_bwd_dp = overlap(iv(dp),iv(math))
    already_tp = overlap(iv(dp), iv(tp))
    tail = max(0,rs_end-end_deps)
    # No native gradient/allreduce dependency is altered by candidate Replay.
    # The final bucket's service is downstream of the layer-1 gradient.
    chosen_ordinal = 34 if "Clate" in case.name else 35
    dispatch = sorted([x for x in rows if x["kind"]=="grad_sync"],key=lambda x:x["cpu_start"])
    selected, = [k for k in dispatch[chosen_ordinal]["kernels"] if nccl(k)]
    bound_overlap = ns(overlap(iv([selected]),iv(qkv["kernels"])))
    ordinary = [k for k in step["kernels"] if not nccl(k)]
    model_inter = ns(overlap(iv(model),iv(ordinary)))
    tp_last_us = [(k["end"]-k["start"])/1e3 for k in tp if k["operation"]=="allreduce"]
    result = dict(rank=rank, rs_union_ms=ns(iv(dp))/1e6,
                  rs_union_with_backward_math_ms=ns(grad_bwd_dp)/1e6,
                  rs_union_with_backward_tp_ms=ns(already_tp)/1e6,
                  rs_union_with_backward_math_or_tp_ms=ns(overlap(iv(dp),iv(math+tp)))/1e6,
                  terminal_rs_wall_tail_ms=tail/1e6,
                  last_rs_service_ms=(dp[-1]["end"]-dp[-1]["start"])/1e6,
                  last_rs_start_after_qkv_ms=(dp[-1]["start"]-qend)/1e6,
                  exposed_target_bucket_tail_ms=max(0,selected["end"]-max(end_deps,selected["start"]))/1e6,
                  qkv_math_ms=ns(iv(qkv["kernels"]))/1e6,
                  qkv_gpu_span_ms=(qend-qstart)/1e6,
                  qkv_host_body_ms=(qkv["cpu_end"]-qkv["cpu_start"])/1e6,
                  qkv_api_entry_to_gpu_ms=(qstart-qkv["cpu_start"])/1e6,
                  qkv_internal_gap_ms=((qend-qstart)-ns(iv(qkv["kernels"])))/1e6,
                  qkv_dp_actual_ms=bound_overlap/1e6,
                  qkv_tp_actual_ms=ns(overlap(iv(qkv["kernels"]),iv(model)))/1e6,
                  target_bucket_duration_ms=(selected["end"]-selected["start"])/1e6,
                  target_rs_after_qkv_starts_ms=max(0,selected["end"]-max(qstart,selected["start"]))/1e6,
                  target_rs_host_submitted_before_qkv_launch=dispatch[chosen_ordinal]["cpu_end"]<=qkv["cpu_start"],
                  optimize_first_gpu_after_rs_ms=(opt_start-rs_end)/1e6 if opt_start else None,
                  optimize_first_compute_after_rs_ms=(opt_math-rs_end)/1e6 if opt_math else None,
                  grad_finish_host_union_ms=ns((x["cpu_start"],x["cpu_end"]) for x in waits)/1e6,
                  tp_total_ms=ns(iv(model))/1e6,
                  tp_overlap_compute_ms=model_inter/1e6,
                  tp_no_compute_ms=(ns(iv(model))-model_inter)/1e6,
                  tp_final_bwd_ar_kernel_us=stat(tp_last_us),
                  final_backward_compute_end_ns=end_math, final_backward_with_tp_end_ns=end_deps,
                  last_rs_end_ns=rs_end, first_opt_gpu_ns=opt_start,
                  step_profiler_gpu_envelope_ms=(max(k["end"] for k in step["kernels"])
                                                -min(k["start"] for k in step["kernels"]))/1e6)
    ranks.append(result)
global_tail=(max(x["last_rs_end_ns"] for x in ranks)-max(x["final_backward_with_tp_end_ns"] for x in ranks))/1e6
fields={k:stat(x[k] for x in ranks) for k,v in ranks[0].items() if
        k!="rank" and isinstance(v,(int,float)) and not isinstance(v,bool) and not k.endswith("_ns")}
out=dict(case=case.name, ranks=ranks,summary=fields,
         cross_rank_wall_tail_ms=max(0,global_tail))
target = Path(__file__).parent/(case.name+"_exposure.json")
target.write_text(json.dumps(out,indent=2))
print(case.name, "global tail",round(global_tail,4))
for k in ("rs_union_ms","rs_union_with_backward_math_or_tp_ms","terminal_rs_wall_tail_ms",
          "last_rs_service_ms","exposed_target_bucket_tail_ms","qkv_math_ms","qkv_internal_gap_ms",
          "qkv_dp_actual_ms","target_rs_after_qkv_starts_ms","tp_total_ms","tp_overlap_compute_ms",
          "optimize_first_gpu_after_rs_ms"):
    print(k,fields.get(k))
