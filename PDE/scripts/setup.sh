#!/usr/bin/env bash
set -euo pipefail

pde_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$pde_root/.venv"
python_bin="${PYTHON:-python3.10}"

"$python_bin" -c 'import sys; sys.exit(None if sys.version_info[:2] == (3, 10) else "PDE setup requires Python 3.10")'
if [[ "$(uname -s)" != Linux ]]; then
    echo "PDE inference setup requires Linux with an NVIDIA CUDA GPU." >&2
    exit 1
fi

"$python_bin" -m venv "$venv"
"$venv/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
"$venv/bin/python" -m pip install -r "$pde_root/requirements.txt"
"$venv/bin/python" "$pde_root/transformers_compat.py"
"$venv/bin/python" -m nltk.downloader -d "$venv/nltk_data" punkt_tab
