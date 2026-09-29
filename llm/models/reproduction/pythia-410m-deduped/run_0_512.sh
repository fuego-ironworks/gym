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
hf_repo="${PYTHIA_HF_REPO:-}"
hf_revision="${PYTHIA_HF_REVISION:-main}"
remote_prefix="${PYTHIA_REMOTE_PREFIX:-dense-0-512}"

base_config="$repo_root/llm/models/upstream/pythia/pythia-410m-deduped.yml"
derived_config="$run_dir/pythia-410m-deduped-vast.yml"

for required in \
    "$neox_dir/deepy.py" \
    "$data_prefix.bin" \
    "$data_prefix.idx" \
    "$tokenizer" \
    "$base_config"
do
    if [[ ! -e "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done

mkdir -p "$run_dir" "$save_dir"

python3 "$here/make_dense_config.py" \
    --base-config "$base_config" \
    --data-prefix "$data_prefix" \
    --tokenizer "$tokenizer" \
    --save-dir "$save_dir" \
    --microbatch "$microbatch" \
    --checkpoint-stride "$stride" \
    --output "$derived_config"

{
    echo "started_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gym_commit $(git -C "$repo_root" rev-parse HEAD)"
    echo "neox_commit $(git -C "$neox_dir" rev-parse HEAD)"
    echo "neox_diff_sha256 $(git -C "$neox_dir" diff | sha256sum | awk '{print $1}')"
    echo "base_config_sha256 $(sha256sum "$base_config" | awk '{print $1}')"
    echo "derived_config_sha256 $(sha256sum "$derived_config" | awk '{print $1}')"
    echo "dataset_bin_sha256 $(sha256sum "$data_prefix.bin" | awk '{print $1}')"
    echo "dataset_idx_sha256 $(sha256sum "$data_prefix.idx" | awk '{print $1}')"
    echo "tokenizer_sha256 $(sha256sum "$tokenizer" | awk '{print $1}')"
    echo "microbatch $microbatch"
    echo "gradient_accumulation $((1024 / microbatch))"
    echo "checkpoint_stride $stride"
    if [[ -n "$hf_repo" ]]; then
        echo "checkpoint_repo $hf_repo"
        echo "checkpoint_revision $hf_revision"
        echo "checkpoint_remote_prefix $remote_prefix"
    else
        echo "checkpoint_repo none"
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$run_dir/manifest.txt"

echo
echo "Nominal train-iters remains 143000."
echo "Derived config exits at step 512 after checkpointing."
if [[ -n "$hf_repo" ]]; then
    echo "Ready checkpoints will be archived, hash-verified on Hugging Face, then retired locally."
elif [[ "$stride" == "1" ]]; then
    echo "WARNING: stride 1 without PYTHIA_HF_REPO retains every checkpoint locally." >&2
fi
echo

upload_pid=""
train_pid=""
cleanup() {
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
        kill "$train_pid" 2>/dev/null || true
    fi
    if [[ -n "$upload_pid" ]] && kill -0 "$upload_pid" 2>/dev/null; then
        kill "$upload_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ -n "$hf_repo" ]]; then
    python3 "$here/checkpoint_retire.py" \
        --save-dir "$save_dir" \
        --receipts-dir "$run_dir/checkpoint-receipts" \
        --staging-dir "$run_dir/upload-staging" \
        --repo "$hf_repo" \
        --revision "$hf_revision" \
        --remote-prefix "$remote_prefix" \
        --stop-step 512 \
        > >(tee "$run_dir/checkpoint-upload.log") 2>&1 &
    upload_pid=$!
fi

cd "$neox_dir"
python3 deepy.py train.py "$derived_config" \
    > >(tee "$run_dir/train.log") 2>&1 &
train_pid=$!

# A dead upload worker during a stride-1 run can silently turn into hundreds of
# gigabytes of retained checkpoints. Stop training instead of discovering that
# failure only after the disk fills.
while kill -0 "$train_pid" 2>/dev/null; do
    if [[ -n "$upload_pid" ]] && ! kill -0 "$upload_pid" 2>/dev/null; then
        set +e
        wait "$upload_pid"
        upload_status=$?
        set -e
        if [[ "$upload_status" -ne 0 ]]; then
            echo "checkpoint upload worker failed; stopping training" >&2
            kill "$train_pid" 2>/dev/null || true
            wait "$train_pid" 2>/dev/null || true
            exit "$upload_status"
        fi
        upload_pid=""
    fi
    sleep 2
done

set +e
wait "$train_pid"
train_status=$?
set -e
train_pid=""
if [[ "$train_status" -ne 0 ]]; then
    echo "training failed with status $train_status" >&2
    exit "$train_status"
fi

if [[ -n "$upload_pid" ]]; then
    wait "$upload_pid"
    upload_pid=""
fi

if [[ -n "$hf_repo" ]]; then
    receipt="$run_dir/checkpoint-receipts/step-000512.json"
    if [[ ! -f "$receipt" ]]; then
        echo "training exited without a verified step-512 upload receipt" >&2
        exit 1
    fi
    echo "step-512 checkpoint uploaded and hash-verified: $receipt"
else
    if [[ ! -d "$save_dir/global_step512" ]]; then
        echo "training exited without a global_step512 checkpoint" >&2
        exit 1
    fi
    echo "step-512 checkpoint present: $save_dir/global_step512"
fi

echo "run complete; inspect verification receipts before destroying the rental"
trap - EXIT INT TERM
