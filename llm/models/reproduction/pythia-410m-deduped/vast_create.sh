#!/usr/bin/env bash
set -euo pipefail

offer_id="${1:?usage: vast_create.sh OFFER_ID}"
disk_gb="${VAST_DISK_GB:-600}"
image="${VAST_IMAGE:-nvidia/cuda:11.1.1-devel-ubuntu20.04}"
label="${VAST_LABEL:-pythia-410m-early-checkpoints}"

cat <<EOF
About to rent Vast.ai offer: $offer_id
disk:  $disk_gb GB
image: $image
label: $label
EOF

if [[ "${VAST_CREATE:-0}" != "1" ]]; then
    echo
    echo "dry run only"
    echo "Set VAST_CREATE=1 to execute the rental."
    echo
    echo "vastai create instance $offer_id --image $image --disk $disk_gb --label $label"
    exit 0
fi

vastai create instance "$offer_id" \
    --image "$image" \
    --disk "$disk_gb" \
    --label "$label"
