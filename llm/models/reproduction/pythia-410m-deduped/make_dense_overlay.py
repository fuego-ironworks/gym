#!/usr/bin/env python3
"""Generate a GPT-NeoX v1.0 overlay for a bounded Pythia-410M reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORIGINAL_SEQUENCES_PER_STEP = 32 * 32
ORIGINAL_TRAIN_ITERS = 143_000
STOP_STEP = 512
OFFICIAL_EARLY_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def build_overlay(
    *,
    data_prefix: str,
    tokenizer: str,
    save_dir: str,
    microbatch: int,
    checkpoint_stride: int,
    slim_checkpoints: bool,
) -> dict:
    if microbatch <= 0:
        raise ValueError("microbatch must be positive")
    if ORIGINAL_SEQUENCES_PER_STEP % microbatch:
        raise ValueError(
            f"microbatch {microbatch} does not divide "
            f"{ORIGINAL_SEQUENCES_PER_STEP}"
        )
    if checkpoint_stride <= 0 or STOP_STEP % checkpoint_stride:
        raise ValueError(
            f"checkpoint stride must be a positive divisor of {STOP_STEP}"
        )

    gas = ORIGINAL_SEQUENCES_PER_STEP // microbatch
    if microbatch * gas != ORIGINAL_SEQUENCES_PER_STEP:
        raise AssertionError("global batch invariant failed")

    # Keep 143000 nominal iterations. exit-interval stops after checkpointing
    # iteration 512 without changing the LR schedule or nominal data length.
    overlay = {
        "train-iters": ORIGINAL_TRAIN_ITERS,
        "lr-decay-iters": ORIGINAL_TRAIN_ITERS,
        "train_micro_batch_size_per_gpu": microbatch,
        "gas": gas,
        "checkpoint-scale": "linear",
        "checkpoint-factor": checkpoint_stride,
        "extra-save-iters": list(OFFICIAL_EARLY_STEPS),
        "exit-interval": STOP_STEP,
        "save": save_dir,
        "train-data-paths": [data_prefix],
        "valid-data-paths": [data_prefix],
        "test-data-paths": [data_prefix],
        "vocab-file": tokenizer,
        "launcher": "pdsh",
        "deepspeed_slurm": False,
        "no-save-optim": slim_checkpoints,
        "no-save-rng": slim_checkpoints,
        # Avoid periodic validation before our deliberate early exit.
        "eval-interval": ORIGINAL_TRAIN_ITERS,
    }
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--checkpoint-stride", type=int, default=8)
    parser.add_argument("--full-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    overlay = build_overlay(
        data_prefix=args.data_prefix,
        tokenizer=args.tokenizer,
        save_dir=args.save_dir,
        microbatch=args.microbatch,
        checkpoint_stride=args.checkpoint_stride,
        slim_checkpoints=not args.full_checkpoints,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gas = overlay["gas"]
    print(
        f"wrote {args.output}: microbatch={args.microbatch}, "
        f"gas={gas}, global_sequences={args.microbatch * gas}, "
        f"stop={STOP_STEP}"
    )


if __name__ == "__main__":
    main()
