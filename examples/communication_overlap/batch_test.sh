#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch test for the communication-overlap demo.
#
# Runs every supported collective against a sweep of communication sizes so
# you can see how overlap effectiveness changes with message size.
#
# Usage:
#   NUM_GPUS=2 bash examples/communication_overlap/batch_test.sh
#
#   # Override sweep ranges:
#   COMM_SIZES_MB="32 128 512" MATRIX_SIZE=8192 bash .../batch_test.sh
#
#   # Quick smoke test:
#   COMM_SIZES_MB="64 128" ITERATIONS=10 bash .../batch_test.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# --- tunable knobs (all overridable via environment) ---
NUM_GPUS="${NUM_GPUS:-2}"                       # GPU count
COLLECTIVES=("all_reduce" "all_gather")          # all supported collectives
COMM_SIZES_MB="${COMM_SIZES_MB:-16 64 128 256}" # message sizes (MB) — README sweep
MATRIX_SIZE="${MATRIX_SIZE:-4096}"               # GEMM dimension
COMPUTE_REPEATS="${COMPUTE_REPEATS:-8}"          # GEMM invocations per measurement
ITERATIONS="${ITERATIONS:-30}"                   # measurement rounds (excl. warmup)
WARMUP="${WARMUP:-5}"                            # warmup rounds
DTYPE="${DTYPE:-bf16}"                           # bf16 | fp32
PYTHON="${PYTHON:-python3}"
HPCX_ROOT="${HPCX_ROOT:-/opt/hpcx}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SH="${SCRIPT_DIR}/run.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/batch_results_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
red()    { echo -e "\033[31m$*\033[0m"; }
green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
bold()   { echo -e "\033[1m$*\033[0m"; }

die() {
    red "FATAL: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# pre-flight
# ---------------------------------------------------------------------------
command -v "${PYTHON}" >/dev/null 2>&1 || die "Python interpreter '${PYTHON}' not found."

if [[ ! -f "${RUN_SH}" ]]; then
    die "run.sh not found at ${RUN_SH}"
fi

# Apply the same HPC-X UCC/UCX LD_LIBRARY_PATH fix that run.sh uses, so that
# import torch succeeds during the pre-flight GPU checks below.
if [[ -d "${HPCX_ROOT}/ucc/lib" && -d "${HPCX_ROOT}/ucx/lib" ]]; then
    CLEAN_LIBRARY_PATH=""
    IFS=: read -ra LIBRARY_DIRS <<< "${LD_LIBRARY_PATH:-}"
    for library_dir in "${LIBRARY_DIRS[@]}"; do
        [[ -z "${library_dir}" ]] && continue
        [[ "${library_dir}" == "${HPCX_ROOT}/ucc/lib" ]] && continue
        [[ "${library_dir}" == "${HPCX_ROOT}/ucx/lib" ]] && continue
        CLEAN_LIBRARY_PATH="${CLEAN_LIBRARY_PATH:+${CLEAN_LIBRARY_PATH}:}${library_dir}"
    done
    export LD_LIBRARY_PATH="${HPCX_ROOT}/ucx/lib:${HPCX_ROOT}/ucc/lib${CLEAN_LIBRARY_PATH:+:${CLEAN_LIBRARY_PATH}}"
fi

# Three-step GPU check with clear diagnostics for each failure mode.
if ! "${PYTHON}" -c "import torch" 2>/tmp/batch_test_torch_import.err; then
    cat /tmp/batch_test_torch_import.err >&2
    die "'${PYTHON}' cannot import torch.  Set PYTHON= to a PyTorch-enabled interpreter."
fi

_gpu_count=$("${PYTHON}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [[ -z "${_gpu_count}" || "${_gpu_count}" -eq 0 ]]; then
    die "torch.cuda.device_count() returned ${_gpu_count:-0}.  Check CUDA driver / GPU visibility."
fi

if [[ "${_gpu_count}" -lt "${NUM_GPUS}" ]]; then
    die "Requested ${NUM_GPUS} GPUs but only ${_gpu_count} are visible to PyTorch."
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------
cat <<EOF

$(bold "==============================================")
$(bold "  Communication Overlap — Batch Test")
$(bold "==============================================")
  GPUs:          ${NUM_GPUS}
  Collectives:   ${COLLECTIVES[*]}
  Comm sizes:    ${COMM_SIZES_MB} MB
  Matrix size:   ${MATRIX_SIZE}
  Compute reps:  ${COMPUTE_REPEATS}
  Iterations:    ${ITERATIONS}  (warmup: ${WARMUP})
  dtype:         ${DTYPE}
  Output dir:    ${OUTPUT_DIR}
$(bold "==============================================")

EOF

# ---------------------------------------------------------------------------
# quick validation pass (--validate-only, single combination)
# ---------------------------------------------------------------------------
echo "$(bold '[validate]') Running correctness checks before batch..."

NUM_GPUS="${NUM_GPUS}" \
    PYTHON="${PYTHON}" \
    HPCX_ROOT="${HPCX_ROOT}" \
    bash "${RUN_SH}" \
    --collective all_reduce \
    --comm-size-mb 64 \
    --matrix-size "${MATRIX_SIZE}" \
    --compute-repeats "${COMPUTE_REPEATS}" \
    --dtype "${DTYPE}" \
    --validate-only \
    >"${OUTPUT_DIR}/validate.log" 2>&1 ||
    die "Validation failed (see ${OUTPUT_DIR}/validate.log)."

green "[validate] OK"
echo ""

# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------
TOTAL=$(( ${#COLLECTIVES[@]} * $(echo "${COMM_SIZES_MB}" | wc -w) ))
CURRENT=0
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

{
    printf "%-14s %10s %12s %12s %12s %12s %12s\n" \
        "collective" "comm_mb" "compute_ms" "comm_ms" "serial_ms" "overlap_ms" "speedup"
    printf "%s\n" "$(printf '=%.0s' {1..88})"
} >"${SUMMARY_FILE}"

for collective in "${COLLECTIVES[@]}"; do
    for comm_size in ${COMM_SIZES_MB}; do
        CURRENT=$((CURRENT + 1))
        label="${collective} / ${comm_size} MB"
        log_file="${OUTPUT_DIR}/${collective}_${comm_size}MB.log"

        echo "$(bold "[${CURRENT}/${TOTAL}]") ${label}"

        NUM_GPUS="${NUM_GPUS}" \
            PYTHON="${PYTHON}" \
            HPCX_ROOT="${HPCX_ROOT}" \
            bash "${RUN_SH}" \
            --collective "${collective}" \
            --comm-size-mb "${comm_size}" \
            --matrix-size "${MATRIX_SIZE}" \
            --compute-repeats "${COMPUTE_REPEATS}" \
            --warmup "${WARMUP}" \
            --iterations "${ITERATIONS}" \
            --dtype "${DTYPE}" \
            >"${log_file}" 2>&1 ||
            die "Run failed (see ${log_file})."

        # Extract key numbers (use Python for reliable regex parsing).
        _extract() {
            "${PYTHON}" -c "
import re, sys
m = re.search(r'${1}', open(sys.argv[1]).read(), re.MULTILINE)
if m:
    print(m.group(1))
else:
    print('NA')
" "${log_file}"
        }
        compute_med=$(_extract '^Compute only\s+([0-9.]+)')
        comm_med=$(_extract    '^Communication only\s+([0-9.]+)')
        serial_med=$(_extract  '^Serial \(sync API\)\s+([0-9.]+)')
        overlap_med=$(_extract '^Delayed-wait overlap\s+([0-9.]+)')
        speedup=$(_extract     'Paired early-wait/overlap speedup:\s+([0-9.]+)x')

        printf "%-14s %10s %12s %12s %12s %12s %12s\n" \
            "${collective}" "${comm_size}" "${compute_med}" "${comm_med}" \
            "${serial_med}" "${overlap_med}" "${speedup}" \
            >>"${SUMMARY_FILE}"

        green "  speedup: ${speedup}x  |  overlap: ${overlap_med} ms  |  serial: ${serial_med} ms"
        echo ""
    done
done

# ---------------------------------------------------------------------------
# final report
# ---------------------------------------------------------------------------

# Also create a stable symlink so the latest summary is always at a fixed path.
LATEST_SUMMARY="${SCRIPT_DIR}/batch_summary.txt"
ln -sf "${SUMMARY_FILE}" "${LATEST_SUMMARY}"

cat <<EOF
$(bold "==============================================")
$(bold "  Batch complete — ${CURRENT}/${TOTAL} runs")
$(bold "==============================================")

Summary table:  ${SUMMARY_FILE}
  (also:        ${LATEST_SUMMARY})
Full logs:      ${OUTPUT_DIR}/
EOF

echo ""
echo "Summary:"
cat "${SUMMARY_FILE}"
