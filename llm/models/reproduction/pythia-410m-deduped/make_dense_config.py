#!/usr/bin/env python3
"""Derive one complete GPT-NeoX v1.0 config for the bounded Pythia run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ORIGINAL_SEQUENCES_PER_STEP = 32 * 32
ORIGINAL_TRAIN_ITERS = 143_000
ORIGINAL_SEED = 1234
STOP_STEP = 1000
OFFICIAL_EARLY_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)


def checkpoint_schedule(stride=8):
    if stride <= 0 or (STOP_STEP - 512) % stride:
        raise ValueError('stride must divide the 512–1000 interval')
    return sorted(set(OFFICIAL_EARLY_STEPS) | set(range(512, STOP_STEP + 1, stride)))


def require_original(config: dict) -> None:
    expected = {
        "train-iters": ORIGINAL_TRAIN_ITERS,
        "lr-decay-iters": ORIGINAL_TRAIN_ITERS,
        "train_micro_batch_size_per_gpu": 32,
        "gas": 1,
        "seq-length": 2048,
        "checkpoint-factor": 1000,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if actual != value:
            raise ValueError(
                f"refusing to derive from unexpected upstream config: "
                f"{key}={actual!r}, expected {value!r}"
            )


def build_config(
    base_config: Path,
    *,
    data_prefix: str,
    tokenizer: str,
    save_dir: str,
    microbatch: int,
    checkpoint_stride: int,
    slim_checkpoints: bool,
) -> dict:
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("base config did not parse to a mapping")
    require_original(config)

    if microbatch <= 0:
        raise ValueError("microbatch must be positive")
    if ORIGINAL_SEQUENCES_PER_STEP % microbatch:
        raise ValueError(
            f"microbatch {microbatch} does not divide "
            f"{ORIGINAL_SEQUENCES_PER_STEP}"
        )
    schedule = checkpoint_schedule(checkpoint_stride)

    gas = ORIGINAL_SEQUENCES_PER_STEP // microbatch

    config.update(
        {
            # Make the NeoX default explicit in the derived receipt.
            "seed": ORIGINAL_SEED,
            "train_micro_batch_size_per_gpu": microbatch,
            "gas": gas,
            "checkpoint-scale": "linear",
            "checkpoint-factor": 1000,
            "extra-save-iters": schedule,
            # NeoX checks this after the checkpoint save in training.py.
            "exit-interval": STOP_STEP,
            "save": save_dir,
            "train-data-paths": [data_prefix],
            "valid-data-paths": [data_prefix],
            "test-data-paths": [data_prefix],
            "vocab-file": tokenizer,
            "launcher": "pdsh",
            "deepspeed_slurm": False,
            "num_gpus": 1,
            "no-save-optim": slim_checkpoints,
            "no-save-rng": slim_checkpoints,
        }
    )

    # Never change these merely to make the rental shorter. They define the
    # original LR schedule and nominal training stream.
    if config["train-iters"] != ORIGINAL_TRAIN_ITERS:
        raise AssertionError("train-iters changed")
    if config["lr-decay-iters"] != ORIGINAL_TRAIN_ITERS:
        raise AssertionError("lr-decay-iters changed")
    if microbatch * gas != ORIGINAL_SEQUENCES_PER_STEP:
        raise AssertionError("global batch invariant failed")

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--checkpoint-stride", type=int, default=8)
    parser.add_argument("--full-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = build_config(
        args.base_config,
        data_prefix=args.data_prefix,
        tokenizer=args.tokenizer,
        save_dir=args.save_dir,
        microbatch=args.microbatch,
        checkpoint_stride=args.checkpoint_stride,
        slim_checkpoints=not args.full_checkpoints,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    gas = config["gas"]
    print(
        f"wrote {args.output}: microbatch={args.microbatch}, "
        f"gas={gas}, global_sequences={args.microbatch * gas}, "
        f"stop={STOP_STEP}"
    )


if __name__ == "__main__":
    main()
