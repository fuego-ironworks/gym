#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work_root="${PYTHIA_WORK_ROOT:-/workspace/pythia-early}"
neox_dir="$work_root/gpt-neox"
tokenizer="$work_root/20B_tokenizer.json"
pythia_commit="a19eecb807ec2c79a39ebf18108816e6ffffc1d5"
slim_patch="$here/neox-v1-slim-checkpoints.patch"

mkdir -p "$work_root"

echo "GPU:"
nvidia-smi || {
    echo "NVIDIA GPU is not visible" >&2
    exit 1
}

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        pdsh \
        python3 \
        python3-dev \
        python3-pip \
        rsync
fi

python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        f"expected Python 3.8 from the Ubuntu 20.04 image; got {sys.version}"
    )
PY

# Stay close to the historical NeoX v1.0 container rather than asking a 2026
# packaging stack to build 2022 CUDA extensions.
python3 -m pip install --upgrade \
    "pip==22.0.4" \
    "setuptools==59.5.0" \
    "wheel==0.37.1"

python3 -m pip install \
    "torch==1.8.1+cu111" \
    -f https://download.pytorch.org/whl/torch_stable.html

if [[ ! -d "$neox_dir/.git" ]]; then
    git clone https://github.com/EleutherAI/gpt-neox.git "$neox_dir"
fi
git -C "$neox_dir" fetch --tags origin
git -C "$neox_dir" checkout --detach v1.0
git -C "$neox_dir" reset --hard v1.0

python3 -m pip install -r "$neox_dir/requirements/requirements.txt"
python3 -m pip install -r "$neox_dir/requirements/requirements-flashattention.txt"
python3 -m pip install "protobuf==3.20.*" pyyaml

# Pythia's historical NeoX image installed this exact Apex revision. Without
# it NeoX falls back to DeepSpeed FusedAdam, which is a needless source of
# optimizer-level divergence for a reproduction run.
python3 -m pip install \
    -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --global-option="--cpp_ext" \
    --global-option="--cuda_ext" \
    "git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597"

(
    cd "$neox_dir"
    python3 megatron/fused_kernels/setup.py install
)

# v1.0 exposes no_save_optim but its DeepSpeed save still emits optimizer
# shards. Apply the pinned, analysis-only cleanup after each collective save.
if git -C "$neox_dir" apply --check "$slim_patch" 2>/dev/null; then
    git -C "$neox_dir" apply "$slim_patch"
elif git -C "$neox_dir" apply --reverse --check "$slim_patch" 2>/dev/null; then
    echo "slim-checkpoint patch already applied"
else
    echo "NeoX source does not match the pinned slim-checkpoint patch" >&2
    exit 1
fi

if [[ ! -f "$tokenizer" ]]; then
    curl --fail --location \
        "https://raw.githubusercontent.com/EleutherAI/pythia/$pythia_commit/utils/20B_tokenizer.json" \
        --output "$tokenizer"
fi

{
    echo "gpt-neox $(git -C "$neox_dir" rev-parse HEAD)"
    echo "gpt-neox_diff_sha256 $(git -C "$neox_dir" diff | sha256sum | awk '{print $1}')"
    echo "python $(python3 --version 2>&1)"
    echo "torch $(python3 -c 'import torch; print(torch.__version__)')"
    echo "tokenizer_sha256 $(sha256sum "$tokenizer" | awk '{print $1}')"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$work_root/bootstrap-manifest.txt"

echo
echo "bootstrap complete"
echo "NeoX:      $neox_dir"
echo "tokenizer: $tokenizer"
