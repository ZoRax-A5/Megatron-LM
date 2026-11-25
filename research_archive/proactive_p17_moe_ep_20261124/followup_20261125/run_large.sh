#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/instrumented_entry.py
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_followup_20261125
for mode in baseline fc1; do
 for run in 1 2 3; do
  export P17_OUTPUT="$R/large_${mode}_${run}"
  extra=(); [ "$mode" = fc1 ] && extra=(--recompute-granularity selective --recompute-modules moe_fc1)
  research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --hidden-size 1024 --ffn-hidden-size 4096 --num-attention-heads 16 --train-iters 30 "${extra[@]}" > "$P17_OUTPUT.log" 2>&1
 done
done
