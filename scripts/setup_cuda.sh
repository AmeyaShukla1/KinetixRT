#!/usr/bin/env bash
# Bare-metal setup for Ubuntu 22.04/24.04 with an NVIDIA GPU (target: RTX 4090, 24 GB).
# NOT yet executed on GPU hardware by the authors - see PROJECT_STATE.md.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: install the NVIDIA driver first (ubuntu-drivers autoinstall)" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

sudo apt-get update
sudo apt-get install -y --no-install-recommends git curl build-essential cmake libegl1 libgl1 libgles2 libglvnd0 libosmesa6 ffmpeg
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements-cuda.txt
uv pip install -e . --no-deps

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
print("device:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY
echo "OK. Next: make gpu-check && make gpu-benchmark"
