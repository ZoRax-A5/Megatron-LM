#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/gate_a_20261126/entry.py NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
for mode in reference3 proactive3; do
 export P17_MOE_PROACTIVE=0; [ "$mode" = proactive3 ] && export P17_MOE_PROACTIVE=1
 export P17_VERIFY_OUT=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_gate_a_20261126/$mode
 research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --train-iters 3 --deterministic-mode --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_VERIFY_OUT.log" 2>&1
done
