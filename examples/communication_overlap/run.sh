#!/usr/bin/env bash

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
PYTHON="${PYTHON:-python3}"
HPCX_ROOT="${HPCX_ROOT:-/opt/hpcx}"

# Some NVIDIA containers expose HPC-X UCC while an incompatible system UCX
# appears earlier in LD_LIBRARY_PATH. Keep UCC and its UCX dependency from the
# same HPC-X installation to avoid missing-symbol errors during import torch.
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

if ! "${PYTHON}" -c "import torch" 2>/tmp/communication_overlap_torch_import.err; then
    cat /tmp/communication_overlap_torch_import.err >&2
    cat >&2 <<EOF

PyTorch failed to load before the overlap experiment started.
If the error mentions libucc.so and a missing ucs_* symbol, UCC and UCX are
coming from incompatible installations.

Current LD_LIBRARY_PATH:
${LD_LIBRARY_PATH:-<unset>}

Expected matching HPC-X directories:
  ${HPCX_ROOT}/ucx/lib
  ${HPCX_ROOT}/ucc/lib

Set HPCX_ROOT to the matching installation, for example:
  HPCX_ROOT=/opt/hpcx NUM_GPUS=${NUM_GPUS} bash $0
EOF
    exit 1
fi

"${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${NUM_GPUS}" \
    "$(dirname "$0")/overlap_demo.py" \
    "$@"
