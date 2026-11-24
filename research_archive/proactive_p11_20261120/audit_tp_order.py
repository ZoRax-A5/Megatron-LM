"""Check the observed TP-model kernel submission signature against native C0."""
from pathlib import Path
from collections import Counter
import json

root = Path(__file__).resolve().parent
cases = ["S0","C1early","C1late"]
out = {}
ref = {}

def kernel_signature(k):
    return k["name"]

for mode in cases:
    case = root/f"p11_4b_{mode}_trace_conn1"
    marks = json.loads((case/"nvtx_attribution.json").read_text())
    comm = json.loads((case/"comm.json").read_text())
    ranks = []
    for rank in range(8):
        group = [x for x in comm if x["kind"]=="tp_model" and x["rank"]==rank]
        by_id = {(x["pid"],x["correlation"],x["start"]): x for x in group}
        fwd_bwd = []
        for scope in sorted((e for e in marks[str(rank)] if e["kind"] in ("forward","backward")),
                            key=lambda e:e["cpu_start"]):
            ordered = sorted((by_id[k["pid"],k["correlation"],k["start"]] for k in scope["kernels"]
                              if (k["pid"],k["correlation"],k["start"]) in by_id),
                             key=lambda k:k["correlation"])
            fwd_bwd.append((scope["stage"],[kernel_signature(k) for k in ordered]))
        if mode == "S0":
            ref[rank] = fwd_bwd
        assert fwd_bwd==ref[rank], (rank,mode)
        ranks.append(dict(rank=rank, count=len(group), phase_counts=dict(
            Counter(x["stage"][0] for x in group)), ordered_names_match=True))
    out[mode] = ranks
(root/"tp_order_audit.json").write_text(json.dumps(out,indent=2))
print("PASS TP kernel submission signature on all 8 ranks")
