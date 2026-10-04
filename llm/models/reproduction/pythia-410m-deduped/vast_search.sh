#!/usr/bin/env bash
set -euo pipefail

disk_gb="${VAST_DISK_GB:-650}"
gpu_ram_gb="${VAST_GPU_RAM_GB:-24}"

query="gpu_ram >= $gpu_ram_gb num_gpus = 1 verified = true rentable = true disk_space >= $disk_gb direct_port_count > 0"

echo "Vast.ai query:"
echo "  $query"
echo
echo "Prefer an Ampere GPU for the legacy GPT-NeoX v1.0 / flash-attn 0.2.2 stack."
echo "This command only lists offers; it does not rent one."
echo

vastai search offers "$query"
