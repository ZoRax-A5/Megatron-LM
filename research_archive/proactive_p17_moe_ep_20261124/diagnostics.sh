#!/usr/bin/env bash
set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/instrumented_entry.py
RESULT=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_20261124
export P17_ROUTING=1
for ep in 2 8; do
 export P17_OUTPUT="$RESULT/ep${ep}_routing"
 research_archive/proactive_p17_moe_ep_20261124/baseline.sh "$ep" --train-iters 3 > "$P17_OUTPUT.log" 2>&1
done
unset P17_ROUTING
export P17_TRACE=1
for ep in 2 8; do
 export P17_OUTPUT="$RESULT/ep${ep}_trace"
 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --output="$P17_OUTPUT" research_archive/proactive_p17_moe_ep_20261124/baseline.sh "$ep" --train-iters 3 > "$P17_OUTPUT.log" 2>&1
done
unset P17_TRACE
export P17_SNAPSHOT=1
for kind in baseline fc1; do
 export P17_OUTPUT="$RESULT/ep2_snapshot_${kind}"
 if [ "$kind" = fc1 ]; then
  research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --train-iters 2 --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_OUTPUT.log" 2>&1
 else
  research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --train-iters 2 > "$P17_OUTPUT.log" 2>&1
 fi
done
unset P17_SNAPSHOT
for run in 1 2 3; do
 export P17_OUTPUT="$RESULT/ep2_fc1_measure${run}"
 research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_OUTPUT.log" 2>&1
done
