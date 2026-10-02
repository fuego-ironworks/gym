#!/usr/bin/env python3
"""Compare one reproduced NeoX checkpoint with the released Pythia anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

OFFICIAL_REPO = "EleutherAI/pythia-410m-deduped"
OFFICIAL_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_PROBE = "The quick brown fox jumps over the lazy dog."


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    prefix = "global_step"
    if not path.name.startswith(prefix) or not path.name[len(prefix):].isdigit():
        raise VerificationError(f"cannot infer step from {path}")
    return int(path.name[len(prefix):])


def tensor_error_stats(local: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    # torch is intentionally duck-typed here so --help and unit tests stay
    # standard-library-only on GitHub-hosted runners.
    local_keys = set(local)
    official_keys = set(official)
    missing = sorted(official_keys - local_keys)
    extra = sorted(local_keys - official_keys)
    if missing or extra:
        raise VerificationError(f"state-dict key mismatch: missing={missing} extra={extra}")

    element_count = 0
    tensor_count = 0
    sum_abs = 0.0
    sum_sq = 0.0
    official_sq = 0.0
    max_abs = 0.0
    worst: list[dict[str, Any]] = []

    for name in sorted(official):
        lhs = local[name].detach().cpu().float()
        rhs = official[name].detach().cpu().float()
        if tuple(lhs.shape) != tuple(rhs.shape):
            raise VerificationError(
                f"shape mismatch for {name}: {tuple(lhs.shape)} != {tuple(rhs.shape)}"
            )
        diff = lhs - rhs
        if not bool(diff.isfinite().all().item()):
            raise VerificationError(f"non-finite difference in {name}")
        count = int(diff.numel())
        tensor_max = float(diff.abs().max().item()) if count else 0.0
        tensor_mean = float(diff.abs().mean().item()) if count else 0.0
        element_count += count
        tensor_count += 1
        sum_abs += float(diff.abs().sum(dtype=__import__("torch").float64).item())
        sum_sq += float(diff.square().sum(dtype=__import__("torch").float64).item())
        official_sq += float(rhs.square().sum(dtype=__import__("torch").float64).item())
        max_abs = max(max_abs, tensor_max)
        worst.append({"name": name, "max_abs": tensor_max, "mean_abs": tensor_mean})

    worst.sort(key=lambda row: row["max_abs"], reverse=True)
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "max_abs": max_abs,
        "mean_abs": sum_abs / element_count if element_count else 0.0,
        "rmse": math.sqrt(sum_sq / element_count) if element_count else 0.0,
        "relative_l2": math.sqrt(sum_sq / official_sq) if official_sq else 0.0,
        "worst_tensors": worst[:20],
    }


def convert_checkpoint(neox_dir: Path, checkpoint_dir: Path, config_file: Path, output_dir: Path) -> None:
    converter = neox_dir / "tools" / "convert_to_hf.py"
    if not converter.is_file():
        raise VerificationError(f"missing NeoX converter: {converter}")
    command = [
        os.environ.get("PYTHON", "python3"),
        str(converter),
        "--input_dir",
        str(checkpoint_dir),
        "--config_file",
        str(config_file),
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


def model_loss(model: Any, input_ids: Any) -> float:
    import torch
    model.eval()
    with torch.no_grad():
        result = model(input_ids=input_ids, labels=input_ids)
    return float(result.loss.item())


def verify(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    step = args.step if args.step is not None else checkpoint_step(args.checkpoint_dir)
    if step not in OFFICIAL_STEPS:
        raise VerificationError(
            f"step {step} is not a released early anchor: {OFFICIAL_STEPS}"
        )
    revision = f"step{step}"

    with tempfile.TemporaryDirectory(prefix=f"pythia-step{step}-hf-") as directory:
        converted = Path(directory)
        convert_checkpoint(args.neox_dir, args.checkpoint_dir, args.config_file, converted)

        tokenizer = AutoTokenizer.from_pretrained(
            args.official_repo, revision=revision
        )
        input_ids = tokenizer(args.probe, return_tensors="pt")["input_ids"]

        local_model = AutoModelForCausalLM.from_pretrained(converted)
        local_loss = model_loss(local_model, input_ids)
        local_state = local_model.state_dict()

        official_model = AutoModelForCausalLM.from_pretrained(
            args.official_repo, revision=revision
        )
        official_loss = model_loss(official_model, input_ids)
        stats = tensor_error_stats(local_state, official_model.state_dict())

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            args.official_repo, revision=revision
        )
        official_commit = getattr(info, "sha", None)

        result = {
            "schema": 1,
            "step": step,
            "checkpoint": str(args.checkpoint_dir),
            "config_sha256": sha256_file(args.config_file),
            "official_repo": args.official_repo,
            "official_revision": revision,
            "official_commit": official_commit,
            "probe": args.probe,
            "local_probe_loss": local_loss,
            "official_probe_loss": official_loss,
            "probe_loss_delta": local_loss - official_loss,
            "tensor_error": stats,
            "structural_match": True,
            "note": "A one-GPU reproduction is expected to diverge numerically from the released 32-GPU trajectory; these are measurements, not a bit-identity claim.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--neox-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--official-repo", default=OFFICIAL_REPO)
    parser.add_argument("--probe", default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify(args)
    stats = result["tensor_error"]
    print(
        f"step {result['step']} structural_match=true "
        f"relative_l2={stats['relative_l2']:.9g} "
        f"max_abs={stats['max_abs']:.9g} "
        f"loss_delta={result['probe_loss_delta']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
