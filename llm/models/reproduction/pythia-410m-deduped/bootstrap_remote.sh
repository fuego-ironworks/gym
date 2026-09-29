#!/usr/bin/env bash
set -euo pipefail

work_root="${PYTHIA_WORK_ROOT:-/workspace/pythia-early}"
neox_dir="$work_root/gpt-neox"
tokenizer="$work_root/20B_tokenizer.json"
pythia_commit="a19eecb807ec2c79a39ebf18108816e6ffffc1d5"

mkdir -p "$work_root"

echo "GPU:"
nvidia-smi || {
    echo "NVIDIA GPU is not visible" >&2
    exit 1
}

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y         build-essential         curl         git         git-lfs         libopenmpi-dev         openmpi-bin         pdsh         python3         python3-dev         python3-pip         rsync
fi

python3 -m pip install --upgrade "pip<24" wheel setuptools

if [[ ! -d "$neox_dir/.git" ]]; then
    git clone https://github.com/EleutherAI/gpt-neox.git "$neox_dir"
fi
git -C "$neox_dir" fetch --tags origin
git -C "$neox_dir" checkout --detach v1.0

python3 -m pip install -r "$neox_dir/requirements/requirements.txt"
python3 -m pip install -r "$neox_dir/requirements/requirements-flashattention.txt"

if [[ ! -f "$tokenizer" ]]; then
    curl --fail --location       "https://raw.githubusercontent.com/EleutherAI/pythia/$pythia_commit/utils/20B_tokenizer.json"       --output "$tokenizer"
fi

{
    echo "gpt-neox $(git -C "$neox_dir" rev-parse HEAD)"
    echo "python $(python3 --version 2>&1)"
    echo "tokenizer_sha256 $(sha256sum "$tokenizer" | awk '{print $1}')"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$work_root/bootstrap-manifest.txt"

echo
echo "bootstrap complete"
echo "NeoX:      $neox_dir"
echo "tokenizer: $tokenizer"
