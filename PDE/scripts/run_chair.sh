#!/usr/bin/env bash
set -euo pipefail

pde_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$pde_root/.venv/bin/python" ]]; then
    if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
        exec "${PYTHON:-python3}" "$pde_root/run_chair.py" "$@"
    fi
    echo "Set up PDE first with bash $pde_root/scripts/setup.sh." >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
exec "$pde_root/.venv/bin/python" -u "$pde_root/run_chair.py" "$@"
