# Pythia 410M deduped: Vast.ai early-checkpoint reproduction

This directory drafts a disposable Vast.ai run for reproducing the early
`EleutherAI/pythia-410m-deduped` trajectory.

Scope stops at optimizer step 512.

## Reproduction rule

Do **not** shorten the upstream training configuration to 512 iterations.

The derived config keeps the original:

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
order, so call its products a dense Pythia reproduction until comparison
against EleutherAI's released anchor checkpoints measures the divergence.

## Files

- `vast_search.sh` lists candidate one-GPU rentals; it never rents one.
- `vast_create.sh OFFER_ID` is dry-run by default and creates only when
  `VAST_CREATE=1`.
- `bootstrap_remote.sh` reconstructs the legacy GPT-NeoX v1.0 environment.
- `prepare_data.sh` reconstructs the deduplicated-Pile mmap one shard at a
  time so source shards and the complete merged file need not coexist.
- `make_dense_config.py` derives one complete NeoX config from the vendored
  upstream 410M config and checks the original batch/schedule invariants.
- `neox-v1-slim-checkpoints.patch` makes NeoX v1.0's existing
  `no-save-optim` setting useful for this run by removing only the
  just-written DeepSpeed optimizer-state shard.
- `run_0_512.sh` records provenance and launches the bounded run.

GPT-NeoX v1.0 rejects duplicate keys spread across multiple config files.
Consequently this workflow generates one complete derived config rather than
trying to override the upstream config with a second overlay.

## Checkpoint density

Default stride is 8, plus the original anchor steps:

    0 1 2 4 8 16 32 64 128 256 512

Use:

    PYTHIA_CHECKPOINT_STRIDE=1

to save every step. The stride-8 default fits much more comfortably beside the
~417 GB deduplicated training mmap. A full every-step run should use more local
disk or add an upload-and-retire stage before launch.

Dense saves are analysis artifacts, not restart points: optimizer and RNG state
are omitted/removed, while the model-state checkpoint is retained.

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
    bash run_0_512.sh

The scripts do not destroy the rental. Verify/copy the retained artifacts and
then destroy it explicitly.

## Disk policy

The old deduplicated Pythia mmap repository is about 417 GB. Upstream's normal
unsharding recipe holds the source shards and a second merged copy.
`prepare_data.sh` downloads one shard, appends it to the final mmap, records a
crash-safe byte boundary, and deletes that shard.

The offer search defaults to 600 GB for the stride-8 run. Every-step retention
requires substantially more unless checkpoints leave the instance while the run
is active.

## Required verification

At released anchor steps, compare the rerun against EleutherAI's published
weights and losses. Retain:

- Gym commit;
- GPT-NeoX v1.0 commit and local patch hash;
- upstream and derived config hashes;
- dataset mmap SHA-256;
- tokenizer hash;
- GPU, driver and CUDA versions;
- microbatch and gradient accumulation;
- tensor-error statistics and loss at anchor checkpoints.

Only after that comparison should any generated checkpoint be described as an
additional Pythia checkpoint rather than a reproduction.
