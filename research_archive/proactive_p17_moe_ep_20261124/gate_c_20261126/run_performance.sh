#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/entry.py NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 P17_CONNECTIONS=2 P17_QUIET=1 P17_SNAPSHOT=0
unset P17_TRACE
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_gate_c_20261126
for repeat in 1 2 3; do
 case "$repeat" in 1) modes='C0 Cnoop C1 C2';; 2) modes='C2 C1 Cnoop C0';; 3) modes='Cnoop C2 C0 C1';; esac
 for mode in $modes; do
  export P17_MODE=$mode P17_OUTPUT="$R/${mode}_r${repeat}" P17_VERIFY_OUT="$R/${mode}_r${repeat}"
  research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/launch.sh 8 --train-iters 50 --deterministic-mode --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_OUTPUT.log" 2>&1
 done
done
