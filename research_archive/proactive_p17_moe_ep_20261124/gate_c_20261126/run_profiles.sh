#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/gate_c_20261126/trace_entry.py NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 P17_CONNECTIONS=2 P17_QUIET=1 P17_SNAPSHOT=0 P17_TRACE=1
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_gate_c_20261126
for spec in C0:1 Cnoop:1 C1:1 C2:1 C2:2 C1:2 C1:3 C2:3; do
 mode=${spec%:*}; repeat=${spec#*:}
 export P17_MODE=$mode P17_OUTPUT="$R/${mode}_profile_r${repeat}" P17_VERIFY_OUT="$R/${mode}_profile_r${repeat}"
 nsys profile --trace=cuda,nvtx --cuda-event-trace=false --sample=none --cpuctxsw=none --force-overwrite=true --output="$P17_OUTPUT" research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/launch.sh 8 --train-iters 23 --deterministic-mode --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_OUTPUT.log" 2>&1
done
