from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_differential import (
    attach_lora,
    run_steps,
    run_test0,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stepwise rank-one LoRA differential tests against a Pythia checkpoint."
    )
    parser.add_argument("--model", default="EleutherAI/pythia-14m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument(
        "--target",
        default="gpt_neox.layers.0.attention.query_key_value",
        help="Fully qualified nn.Linear module to wrap with the adapter.",
    )
    parser.add_argument("--example", default="The moon orbits the Earth.")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm/models/runs/pythia-14m-lora-inspectability"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rank != 1:
        raise SystemExit("The current mechanical acceptance case requires --rank 1")
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model.eval()

    encoded = tokenizer(args.example, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids.shape[-1] < 2:
        raise SystemExit("training example must tokenize to at least two tokens")
    labels = input_ids.clone()

    adapter = attach_lora(
        model,
        args.target,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test0 = run_test0(model, adapter, input_ids, labels, args.output_dir)
    steps = run_steps(
        model,
        adapter,
        input_ids,
        labels,
        args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    manifest = write_run_manifest(
        args.output_dir,
        model_name=args.model,
        revision=args.revision,
        target=args.target,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=args.seed,
        example=args.example,
        test0=test0,
        step_receipts=steps,
    )
    manifest["runtime"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "platform": platform.platform(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(args.output_dir), "passed": manifest["passed"]}))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
