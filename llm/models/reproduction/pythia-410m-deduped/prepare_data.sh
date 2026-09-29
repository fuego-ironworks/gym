#!/usr/bin/env bash
set -euo pipefail

data_dir="${PYTHIA_DATA_DIR:-/workspace/pythia-data}"
repo_url="${PYTHIA_DATA_URL:-https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps/resolve/main}"
prefix="pile_0.87_deduped_text_document"
last_shard=82
expected_bin_sha256="0cd548efd15974d5cca78f9baddbd59220ca675535dcfc0c350087c79f504693"

mkdir -p "$data_dir"
out="$data_dir/$prefix.bin"
idx="$data_dir/$prefix.idx"
state="$data_dir/.unshard-state"
shard_tmp="$data_dir/.current-shard.bin"

completed=-1
completed_size=0

if [[ -f "$state" ]]; then
    read -r completed completed_size < "$state"
    if [[ ! -f "$out" ]]; then
        echo "state exists but $out does not" >&2
        exit 1
    fi
    # A killed append can leave a partial next shard. Roll back to the last
    # committed byte boundary before resuming.
    truncate -s "$completed_size" "$out"
elif [[ -e "$out" ]]; then
    echo "$out exists without resume state; refusing to guess" >&2
    exit 1
else
    : > "$out"
fi

for ((i=completed + 1; i<=last_shard; i++)); do
    name=$(printf "%s-%05d-of-%05d.bin" "$prefix" "$i" "$last_shard")
    echo "downloading shard $i/$last_shard: $name"
    rm -f "$shard_tmp"
    curl         --fail         --location         --retry 12         --retry-all-errors         --continue-at -         "$repo_url/$name?download=true"         --output "$shard_tmp"

    cat "$shard_tmp" >> "$out"
    sync "$out"
    completed_size=$(stat -c '%s' "$out")

    printf '%d %d\n' "$i" "$completed_size" > "$state.tmp"
    mv "$state.tmp" "$state"
    rm -f "$shard_tmp"
done

actual=$(sha256sum "$out" | awk '{print $1}')
if [[ "$actual" != "$expected_bin_sha256" ]]; then
    echo "dataset checksum mismatch" >&2
    echo "expected: $expected_bin_sha256" >&2
    echo "actual:   $actual" >&2
    exit 1
fi

if [[ ! -f "$idx" ]]; then
    curl         --fail         --location         --retry 12         --retry-all-errors         "$repo_url/$prefix.idx?download=true"         --output "$idx.tmp"
    mv "$idx.tmp" "$idx"
fi

sha256sum "$out" "$idx" > "$data_dir/SHA256SUMS"
echo "dataset ready: $data_dir/$prefix"
