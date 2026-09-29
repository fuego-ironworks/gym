#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)

work_root="${PYTHIA_WORK_ROOT:-/workspace/pythia-early}"
neox_dir="${PYTHIA_NEOX_DIR:-$work_root/gpt-neox}"
data_dir="${PYTHIA_DATA_DIR:-/workspace/pythia-data}"
data_prefix="$data_dir/pile_0.87_deduped_text_document"
tokenizer="${PYTHIA_TOKENIZER:-$work_root/20B_tokenizer.json}"
save_dir="${PYTHIA_SAVE_DIR:-$work_root/checkpoints}"
run_dir="${PYTHIA_RUN_DIR:-$work_root/run}"
microbatch="${PYTHIA_MICROBATCH:-8}"
stride="${PYTHIA_CHECKPOINT_STRIDE:-8}"

base_config="$repo_root/llm/models/upstream/pythia/pythia-410m-deduped.yml"
overlay="$run_dir/vast-0-512-overlay.json"

for required in     "$neox_dir/deepy.py"     "$data_prefix.bin"     "$data_prefix.idx"     "$tokenizer"     "$base_config"
do
    if [[ ! -e "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done

mkdir -p "$run_dir" "$save_dir"

python3 "$here/make_dense_overlay.py"     --data-prefix "$data_prefix"     --tokenizer "$tokenizer"     --save-dir "$save_dir"     --microbatch "$microbatch"     --checkpoint-stride "$stride"     --output "$overlay"

{
    echo "started_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gym_commit $(git -C "$repo_root" rev-parse HEAD)"
    echo "neox_commit $(git -C "$neox_dir" rev-parse HEAD)"
    echo "base_config_sha256 $(sha256sum "$base_config" | awk '{print $1}')"
    echo "overlay_sha256 $(sha256sum "$overlay" | awk '{print $1}')"
    echo "dataset_bin_sha256 $(sha256sum "$data_prefix.bin" | awk '{print $1}')"
    echo "dataset_idx_sha256 $(sha256sum "$data_prefix.idx" | awk '{print $1}')"
    echo "tokenizer_sha256 $(sha256sum "$tokenizer" | awk '{print $1}')"
    echo "microbatch $microbatch"
    echo "checkpoint_stride $stride"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$run_dir/manifest.txt"

echo
echo "Nominal train-iters remains 143000."
echo "The overlay uses exit-interval=512, which NeoX v1.0 checks after saving."
echo

cd "$neox_dir"

# deepy forwards launcher arguments to DeepSpeed. One GPU is intentional; the
# overlay raises gradient accumulation so one optimizer step still contains the
# original 1024 sequences.
python3 deepy.py     train.py     "$base_config"     "$overlay"     --num_gpus 1     2>&1 | tee "$run_dir/train.log"

echo "training process exited"
echo "verify that a step-512 checkpoint exists before retiring the instance"
