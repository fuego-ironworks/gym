# Pythia 410M deduped: Vast.ai early-checkpoint reproduction

This directory runs a disposable Vast.ai reproduction of the early
`EleutherAI/pythia-410m-deduped` trajectory. Scope stops at optimizer step 512.

## Reproduction rule

Do **not** shorten the upstream training configuration to 512 iterations. The
derived config keeps the original:

- `train-iters = 143000`
- `lr-decay-iters = 143000`
- seed 1234
- sequence length 2048
- 1024 sequences / 2,097,152 tokens per optimizer step

It adds `exit-interval = 512`. GPT-NeoX v1.0 checks that condition after its
checkpointing block, so the process exits immediately after step 512 is saved.

On one GPU:

    microbatch * gradient_accumulation = 1024 sequences

The original run used 32 GPUs. A one-GPU rerun changes floating-point reduction
order, so call its products a dense Pythia reproduction. `verify_anchor.py`
measures the resulting divergence against released anchor checkpoints; it does
not assume bit identity.

## Files

- `vast_search.sh` lists candidate one-GPU rentals; it never rents one.
- `vast_create.sh OFFER_ID` is dry-run by default and creates only when
  `VAST_CREATE=1`.
- `bootstrap_remote.sh` reconstructs the legacy GPT-NeoX v1.0 environment.
- `prepare_data.sh` reconstructs the deduplicated-Pile mmap one shard at a
  time so source shards and the complete merged file need not coexist.
- `make_dense_config.py` derives one complete NeoX config from the vendored
  upstream 410M config and checks the original batch/schedule invariants.
- `neox-v1-slim-checkpoints.patch` removes optimizer shards only after every
  rank finishes saving and then publishes an atomic `.gym-ready` marker.
- `neox-v1-py38-converter.patch` makes NeoX v1.0's upstream HF converter run
  under the pinned Python 3.8 image without changing its conversion logic.
- `checkpoint_retire.py` archives a ready checkpoint, uploads it to an existing
  Hugging Face staging repo, verifies the Hub LFS SHA-256, uploads a receipt,
  and only then deletes the local checkpoint.
- `verify_anchor.py` converts one local NeoX checkpoint with the upstream
  converter and compares every HF state-dict tensor plus a fixed probe loss
  against the released Pythia revision.
- `run_0_512.sh` records provenance, supervises the optional upload worker, and
  launches the bounded run.

GPT-NeoX v1.0 rejects duplicate keys spread across multiple config files.
Consequently this workflow generates one complete derived config rather than
trying to override the upstream config with a second overlay.

## Checkpoint density

Default stride is 8, plus the released early anchor steps:

    0 1 2 4 8 16 32 64 128 256 512

Use:

    PYTHIA_CHECKPOINT_STRIDE=1

to save every step. Dense saves are analysis artifacts, not restart points:
optimizer and RNG state are omitted/removed while model-state files are kept.

For stride 1, set `PYTHIA_HF_REPO` so checkpoints leave the rental as they are
produced instead of accumulating beside the ~417 GB training mmap. The staging
repo must already exist; the scripts never create a repository implicitly.

## Vast.ai workflow

Install/authenticate the Vast CLI locally. Keep API and Hugging Face credentials
out of this repository.

    python -m pip install --upgrade vastai
    vastai set api-key ...

Search and inspect offers:

    bash vast_search.sh

Choose an offer manually. Prefer an Ampere GPU such as RTX 3090, A5000, A6000,
or A100 because the pinned 2022 stack uses CUDA 11.1 and PyTorch 1.8.1.

    bash vast_create.sh OFFER_ID
    VAST_CREATE=1 bash vast_create.sh OFFER_ID
    vastai ssh-url INSTANCE_ID

Clone this branch on the rental and run:

    bash bootstrap_remote.sh
    bash prepare_data.sh

For the default local-retention run:

    bash run_0_512.sh

For every-step upload-and-retire, authenticate Hugging Face and name an existing
private staging repository:

    export HF_TOKEN=...
    export PYTHIA_HF_REPO=OWNER/pythia-410m-dense-staging
    export PYTHIA_CHECKPOINT_STRIDE=1
    bash run_0_512.sh

Optional upload namespace controls:

    PYTHIA_HF_REVISION=main
    PYTHIA_REMOTE_PREFIX=dense-0-512

Each uploaded checkpoint becomes one `global_stepN.tar`. The receipt contains a
SHA-256 for the archive and for every file inside it. Local retirement happens
only after the Hub reports the same LFS SHA-256 for the archive and accepts the
receipt upload. A failed upload, missing LFS hash, or hash mismatch leaves the
local checkpoint intact. If the upload worker dies during training,
`run_0_512.sh` stops training rather than silently filling the rental disk.

The scripts do not destroy the rental. Inspect the receipts and destroy it
explicitly when finished.

## Anchor verifier

`verify_anchor.py` accepts a reproduced NeoX checkpoint at one of the released
early steps `0,1,2,4,8,16,32,64,128,256,512`. It runs NeoX v1.0's own
`tools/convert_to_hf.py`, loads the corresponding Hugging Face revision such as
`step128`, requires exact state-dict key and shape agreement, and records:

- released repository revision and commit;
- tensor count and parameter count;
- maximum and mean absolute tensor error;
- RMSE and relative L2 error;
- the 20 tensors with largest maximum absolute error;
- a fixed-text next-token loss for both models and its delta.

Example:

    python3 verify_anchor.py \
      --checkpoint-dir /workspace/pythia-early/checkpoints/global_step128 \
      --config-file /workspace/pythia-early/run/pythia-410m-deduped-vast.yml \
      --neox-dir /workspace/pythia-early/gpt-neox \
      --output /workspace/pythia-early/run/verify-step128.json

A successful verifier run establishes structural compatibility and measures
numerical divergence. It does not relabel the one-GPU output as an official
Pythia checkpoint.

## Disk policy

The old deduplicated Pythia mmap repository is about 417 GB. Upstream's normal
unsharding recipe holds the source shards and a second merged copy.
`prepare_data.sh` downloads one shard, appends it to the final mmap, records a
crash-safe byte boundary, and deletes that shard.

The offer search defaults to 600 GB for the stride-8 run. With upload-and-retire,
stride 1 needs only the current checkpoint plus one temporary tar archive beyond
the training mmap instead of retaining hundreds of checkpoint copies.

## Non-billing tests

The repository's normal standard-library test job exercises this workflow
without contacting Vast.ai or Hugging Face. `tests/test_pythia_vast_reproduction.py`
uses fake CLIs and a fake Hub to prove that:

- `vast_create.sh` does not invoke `vastai` unless `VAST_CREATE=1`;
- the search path only issues `search offers`;
- the explicit create path issues `create instance` in the test double, not a
  real rental request;
- incomplete checkpoints without `.gym-ready` are ignored;
- optimizer-state shards prevent retirement;
- a remote hash mismatch never deletes the local checkpoint;
- a checkpoint is retired only after a matching archive hash and receipt upload.

Run the same test locally with:

    python3 -m unittest discover -s tests
