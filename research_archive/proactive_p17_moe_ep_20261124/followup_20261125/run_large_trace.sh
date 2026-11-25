#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_TRACE=1 P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/followup_20261125/trace_entry.py P17_OUTPUT=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_followup_20261125/large_ep2_trace
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --output="$P17_OUTPUT" research_archive/proactive_p17_moe_ep_20261124/baseline.sh 2 --hidden-size 1024 --ffn-hidden-size 4096 --num-attention-heads 16 --train-iters 13 > "$P17_OUTPUT.log" 2>&1
