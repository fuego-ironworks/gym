# Pythia 410M deduped: Vast.ai early-checkpoint reproduction

This directory is a first operational draft for reproducing the early
`EleutherAI/pythia-410m-deduped` trajectory on a disposable Vast.ai GPU.

Scope is deliberately bounded to optimizer steps 0 through 512. Nothing here
should train past step 512.

## Reproduction rule

Do **not** shorten the upstream training configuration to 512 iterations.

GPT-NeoX uses the nominal training length in the learning-rate schedule and in
training-data setup. The run therefore keeps the original:

- `train-iters = 143000`
- `lr-decay-iters = 143000`
- seed 1234
- sequence length 2048
- 1024 sequences / 2,097,152 tokens per optimizer step

The overlay sets `exit-interval = 512`. GPT-NeoX v1.0 checks that condition
after checkpointing, so the process exits immediately after the step-512
checkpoint is written.

On one GPU, gradient accumulation is chosen so that:

    microbatch * gradient_accumulation = 1024 sequences

Changing the GPU count and reduction order means a one-GPU run must not be
called bit-identical to EleutherAI's original 32-GPU run until comparison at the
published checkpoints shows what the divergence actually is.

## Files

- `vast_search.sh` prints suitable one-GPU offers. It never rents anything.
- `vast_create.sh OFFER_ID` creates the selected instance.
- `bootstrap_remote.sh` installs the legacy GPT-NeoX v1.0 environment.
- `prepare_data.sh` reconstructs the deduplicated Pile mmap without keeping a
  second full copy of all 83 source shards.
- `make_dense_overlay.py` produces the small NeoX overlay while enforcing the
  original global-batch invariant.
- `run_0_512.sh` records machine/config provenance and launches the bounded run.

The checked-in upstream model configuration remains authoritative:
`llm/models/upstream/pythia/pythia-410m-deduped.yml`.

## Checkpoint density

Default checkpoint stride is 8. The original logarithmic checkpoints are added
explicitly, so the run retains steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256 and
512 as anchors.

Set:

    PYTHIA_CHECKPOINT_STRIDE=1

to save every step. That substantially increases local storage and later upload
volume. A stride-8 first pass is meant to locate the interesting interval; a
second bounded run can then save every step only inside that interval.

Dense checkpoints default to weight/RNG-light saves (`no-save-optim` and
`no-save-rng`) because retaining hundreds of Adam states would dominate the
cost. These checkpoints are analysis artifacts, not restart points.

## Vast.ai workflow

Install and authenticate the Vast CLI locally. Do not put API keys or Hugging
Face tokens in this repository.

    python -m pip install --upgrade vastai
    vastai set api-key ...

Search:

    ./vast_search.sh

Choose an offer manually, then:

    ./vast_create.sh OFFER_ID
    vastai ssh-url INSTANCE_ID

Copy or clone this branch on the instance and run:

    ./bootstrap_remote.sh
    ./prepare_data.sh
    ./run_0_512.sh

The Vast scripts intentionally do not destroy the instance automatically.
Verify the step-512 checkpoint and copy/upload retained artifacts first; then
destroy the rental explicitly.

## Disk policy

The old Pythia deduplicated mmap repository is about 417 GB. The upstream
unsharding recipe normally needs source shards plus a second merged copy.
`prepare_data.sh` instead downloads one shard at a time, appends it to the
final mmap, records a crash-safe byte boundary, and removes the shard.

The search script therefore defaults to 600 GB of local disk for a stride-8
run. Saving every step needs materially more disk unless checkpoints are
uploaded and retired during training.

## Required verification before calling these Pythia checkpoints

At the published anchor steps, compare this run against EleutherAI's released
weights and losses. Record at least:

- exact Gym commit;
- GPT-NeoX tag/commit;
- upstream Pythia config hash;
- generated overlay;
- dataset mmap SHA-256;
- tokenizer source/hash;
- GPU model, driver and CUDA version;
- microbatch and gradient accumulation;
- per-anchor tensor error statistics and loss.

Until that comparison exists, call the outputs a dense Pythia reproduction,
not additional official Pythia checkpoints.
