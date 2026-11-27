#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/entry_serial_control.py NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 P17_CONNECTIONS=2 P17_QUIET=1 P17_SNAPSHOT=0 P17_MODE=C1
unset P17_TRACE
R=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_gate_c_20261126
for repeat in 1 2 3; do
 export P17_OUTPUT="$R/C1serial_r${repeat}" P17_VERIFY_OUT="$R/C1serial_r${repeat}"
 research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/launch.sh 8 --train-iters 50 --deterministic-mode --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_OUTPUT.log" 2>&1
done
