#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_TRACE=1 P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/followup_20261125/trace_entry.py
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_followup_20261125
for mode in ep8 ep2 ep2_fc1; do
 export P17_OUTPUT="$R/${mode}_steady"
 case "$mode" in
  ep8) ep=8; extra=() ;;
  ep2) ep=2; extra=() ;;
  ep2_fc1) ep=2; extra=(--recompute-granularity selective --recompute-modules moe_fc1) ;;
 esac
 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --output="$P17_OUTPUT" research_archive/proactive_p17_moe_ep_20261124/baseline.sh "$ep" --train-iters 23 "${extra[@]}" > "$P17_OUTPUT.log" 2>&1
done
