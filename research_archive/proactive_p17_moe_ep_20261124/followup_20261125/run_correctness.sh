#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/followup_20261125/correctness_entry.py
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_followup_20261125
for mode in base1 base2 fc1 det1 det2 det_fc1; do
 export P17_VERIFY_OUT="$R/$mode"
 unset NCCL_ALGO NVTE_ALLOW_NONDETERMINISTIC_ALGO CUBLAS_WORKSPACE_CONFIG
 extra=()
 case "$mode" in det*) export NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8; extra+=(--deterministic-mode);; esac
 case "$mode" in *fc1) extra+=(--recompute-granularity selective --recompute-modules moe_fc1);; esac
 research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --train-iters 1 "${extra[@]}" > "$P17_VERIFY_OUT.log" 2>&1
done
