#!/usr/bin/env python3
"""Summarize per-rank FC1 checkpoint and Combine-consumer identity audit."""
import json
import sys

root = sys.argv[1]
prefix = sys.argv[2]
reports = []
for rank in range(8):
    call_path = f"{root}/{prefix}.calls.rank{rank}.jsonl"
    consumer_path = f"{root}/{prefix}.consumers.rank{rank}.jsonl"
    calls = [json.loads(line) for line in open(call_path)]
    consumers = [json.loads(line) for line in open(consumer_path)]
    errors = []
    for row in calls:
        if row["metadata"].get("module") != "moe_fc1" or row["metadata"].get("layer_number") != 1 or row["metadata"].get("microbatch") != 0:
            errors.append((row["id"], "unexpected checkpoint identity", row["metadata"]))
        if row["calls"] != {"forward": 1, "prefetch": 1}:
            errors.append((row["id"], "unexpected call counts", row["calls"]))
        if not row["cached_grad_fn_at_consume"] or row["state"] != "consumed":
            errors.append((row["id"], "cached graph not consumed", row["state"]))
        if row["parameter_versions_at_consume"] != row["descriptors"][0]["parameter_versions"]:
            errors.append((row["id"], "parameter version changed before consume"))
        if len(row["descriptors"]) != 2:
            errors.append((row["id"], "missing call descriptor"))
        else:
            a, b = row["descriptors"]
            if a["phase"] != "forward" or b["phase"] != "prefetch":
                errors.append((row["id"], "phase mismatch"))
            if a["inputs"] != b["inputs"]:
                errors.append((row["id"], "FC1 input identity/version mismatch"))
            if not a["rng_saved"] or not b["rng_saved"]:
                errors.append((row["id"], "RNG snapshot missing"))
    if not all(x["same_object"] and x["same_storage_pointer"] for x in consumers):
        errors.append(("combine consumer", "output identity mismatch"))
    token_rows = [d["inputs"][0]["shape"][0] for x in calls for d in x["descriptors"][:1] if d.get("inputs")]
    reports.append({
        "rank": rank,
        "checkpoint_invocations": len(calls),
        "combine_consumers": len(consumers),
        "all_checkpoints_are_layer1_microbatch0_moe_fc1": all(x["metadata"].get("module") == "moe_fc1" and x["metadata"].get("layer_number") == 1 and x["metadata"].get("microbatch") == 0 for x in calls),
        "fc1_input_rows_min_mean_max": [min(token_rows), sum(token_rows) / len(token_rows), max(token_rows)] if token_rows else None,
        "forward_plus_prefetch_only": all(x["calls"] == {"forward": 1, "prefetch": 1} for x in calls),
        "all_cached_graphs_consumed": all(x["cached_grad_fn_at_consume"] and x["state"] == "consumed" for x in calls),
        "all_parameter_versions_stable": all(x["parameter_versions_at_consume"] == x["descriptors"][0]["parameter_versions"] for x in calls),
        "all_fc1_inputs_same_storage_and_version": all(len(x["descriptors"]) == 2 and x["descriptors"][0]["inputs"] == x["descriptors"][1]["inputs"] for x in calls),
        "all_rng_snapshots_present": all(all(d["rng_saved"] for d in x["descriptors"]) for x in calls),
        "combine_output_consumed_by_identity": all(x["same_object"] and x["same_storage_pointer"] for x in consumers),
        "errors": errors,
    })
result = {"prefix": prefix, "ranks": reports, "pass": all(not r["errors"] for r in reports)}
json.dump(result, sys.stdout, indent=2)
print()
