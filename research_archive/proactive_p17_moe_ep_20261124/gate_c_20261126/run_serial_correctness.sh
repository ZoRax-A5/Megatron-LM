#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" P17_ENTRY=research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/entry_serial_control.py NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 P17_CONNECTIONS=2 P17_QUIET=1 P17_MODE=C1 P17_SNAPSHOT=1 P17_VERIFY_OUT=/mnt/si002365wekc/zwx/code/recom/qwen3_bench/results/p17_gate_b_20261126/C1_serial
research_archive/proactive_p17_moe_ep_20261124/gate_b_20261126/launch.sh 8 --train-iters 3 --deterministic-mode --recompute-granularity selective --recompute-modules moe_fc1 > "$P17_VERIFY_OUT.log" 2>&1
