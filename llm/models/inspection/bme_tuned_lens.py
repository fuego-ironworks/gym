#!/usr/bin/env python3
"""Layer-by-layer BME likelihood trajectories through a Tuned Lens.

Modes:
- frozen-final: apply one public final-model lens to step512 and step1000.
  This is a fixed-decoder transfer diagnostic, not checkpoint calibration.
- checkpoint-specific: require separately trained lens artifacts whose saved
  base_model_revision exactly matches each checkpoint.

The score keeps the original experiment's separate prompt/continuation encoding
and teacher-forced mean token log probability.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from tuned_lens import TunedLens
from tuned_lens.load_artifacts import load_lens_artifacts
from tuned_lens.nn.unembed import Unembed
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bme_causal_recheck", Path(__file__).with_name("bme_causal_recheck.py")
)
causal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal)

MODEL = causal.MODEL
PAIRS = causal.PAIRS
CHECKPOINTS = ("step512", "step1000")
PROMPTS = {
    "bme_original": causal.PROMPTS["bme_original"],
    "bme_heldout": causal.PROMPTS["bme_heldout"],
    "unrelated_school_question": causal.PROMPTS["unrelated_school_question"],
}
FINAL_REVISION = "step143000"
PUBLIC_LENS_RESOURCE = MODEL


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tsv(path, headings, rows):
    def safe(value):
        return str(value).replace("\t", "\\t").replace("\n", "\\n")

    body = "\t".join(headings) + "\n"
    body += "\n".join(
        "\t".join(safe(row.get(column, "")) for column in headings)
        for row in rows
    )
    atomic_write(path, body + "\n")


def encode_batch(tokenizer, prompt):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("empty prompt")
    entries = []
    for pair_index, (school, check) in enumerate(PAIRS, 1):
        for kind, continuation in (("school", school), ("check", check)):
            continuation_ids = tokenizer.encode(
                continuation, add_special_tokens=False
            )
            joined = tokenizer.encode(
                prompt + continuation, add_special_tokens=False
            )
            ids = prompt_ids + continuation_ids
            entries.append(
                {
                    "pair": pair_index,
                    "kind": kind,
                    "ids": ids,
                    "start": len(prompt_ids),
                    "targets": continuation_ids,
                    "boundary_agrees": ids == joined,
                }
            )

    max_len = max(len(entry["ids"]) for entry in entries)
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no eos token for masked right padding")
    input_ids = torch.full((len(entries), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(entries), max_len), dtype=torch.long)
    for row, entry in enumerate(entries):
        n = len(entry["ids"])
        input_ids[row, :n] = torch.tensor(entry["ids"], dtype=torch.long)
        attention_mask[row, :n] = 1
    return entries, input_ids, attention_mask


def mean_target_logp(logits, targets):
    target = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, target[:, None])
        .mean()
        .item()
    )


def mean_forward_kl(model_logits, lens_logits):
    log_p = torch.log_softmax(model_logits.float(), dim=-1)
    log_q = torch.log_softmax(lens_logits.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean().item()


def summarize_pairs(rows):
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair"], {})[row["kind"]] = row["mean_logp"]
    margins = {}
    for pair, values in sorted(by_pair.items()):
        margins[pair] = values["school"] - values["check"]
    return margins, sum(margins.values()) / len(margins)


def lens_location(index):
    return "input" if index == 0 else f"block{index - 1}_post"


def first_positive(rows):
    for row in rows:
        if row["aggregate_margin"] > 0:
            return row["location"]
    return None


def first_sustained_sign(rows, positive):
    for index, row in enumerate(rows):
        tail = rows[index:]
        if positive and all(item["aggregate_margin"] > 0 for item in tail):
            return row["location"]
        if not positive and all(item["aggregate_margin"] < 0 for item in tail):
            return row["location"]
    return None


def load_public_frozen_lens():
    config_path, params_path = load_lens_artifacts(
        resource_id=PUBLIC_LENS_RESOURCE
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    final_model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=FINAL_REVISION, torch_dtype=torch.float32
    )
    final_model.eval()
    final_hash = Unembed(final_model).unembedding_hash()
    expected_hash = config.get("unembed_hash")
    if expected_hash and expected_hash != final_hash:
        raise RuntimeError(
            "public lens unembed hash does not match final Pythia checkpoint"
        )
    lens = TunedLens.from_model_and_pretrained(
        final_model,
        lens_resource_id=PUBLIC_LENS_RESOURCE,
        map_location="cpu",
    )
    lens.eval()
    metadata = {
        "lens_resource": PUBLIC_LENS_RESOURCE,
        "lens_params_sha256": sha256_file(params_path),
        "lens_base_model": config.get("base_model_name_or_path", ""),
        "lens_base_model_revision": config.get("base_model_revision", ""),
        "lens_unembed_hash": expected_hash or "",
        "decoder_unembed_hash": final_hash,
        "decoder_revision": FINAL_REVISION,
    }
    del final_model
    gc.collect()
    return lens, metadata


def read_checkpoint_lens(lens_dir, checkpoint):
    config_path = Path(lens_dir) / "config.json"
    params_path = Path(lens_dir) / "params.pt"
    if not config_path.is_file() or not params_path.is_file():
        raise FileNotFoundError(f"missing tuned-lens artifacts in {lens_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL:
        raise RuntimeError(
            f"{checkpoint}: wrong base model in lens config"
        )
    if config.get("base_model_revision") != checkpoint:
        raise RuntimeError(
            f"{checkpoint}: lens revision "
            f"{config.get('base_model_revision')!r} does not match checkpoint"
        )
    return config, params_path


def load_checkpoint_lens(model, lens_dir, checkpoint):
    config, params_path = read_checkpoint_lens(lens_dir, checkpoint)
    current_hash = Unembed(model).unembedding_hash()
    expected_hash = config.get("unembed_hash")
    if expected_hash and expected_hash != current_hash:
        raise RuntimeError(f"{checkpoint}: lens unembed hash mismatch")
    lens = TunedLens.from_model_and_pretrained(
        model, lens_resource_id=str(lens_dir), map_location="cpu"
    )
    lens.eval()
    metadata = {
        "lens_resource": str(lens_dir),
        "lens_params_sha256": sha256_file(params_path),
        "lens_base_model": config.get("base_model_name_or_path", ""),
        "lens_base_model_revision": config.get("base_model_revision", ""),
        "lens_unembed_hash": expected_hash or "",
        "decoder_unembed_hash": current_hash,
        "decoder_revision": checkpoint,
    }
    return lens, metadata


def score_prompt(model, lens, tokenizer, prompt_name, prompt):
    entries, input_ids, attention_mask = encode_batch(tokenizer, prompt)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = output.hidden_states[:-1]
    if len(hidden_states) != len(lens):
        raise RuntimeError(
            f"hidden/lens depth mismatch: {len(hidden_states)} != {len(lens)}"
        )

    pair_rows = []
    trajectory = []
    for layer, hidden in enumerate(hidden_states):
        scored = []
        kls = []
        for row_index, entry in enumerate(entries):
            start = entry["start"]
            stop = start + len(entry["targets"])
            predictor_hidden = hidden[row_index, start - 1 : stop - 1, :]
            model_logits = output.logits[
                row_index, start - 1 : stop - 1, :
            ]
            with torch.inference_mode():
                lens_logits = lens(predictor_hidden, idx=layer)
            value = mean_target_logp(lens_logits, entry["targets"])
            kls.append(mean_forward_kl(model_logits, lens_logits))
            row = {
                "prompt": prompt_name,
                "lens_index": layer,
                "location": lens_location(layer),
                "pair": entry["pair"],
                "kind": entry["kind"],
                "mean_logp": value,
                "boundary_agrees": entry["boundary_agrees"],
            }
            scored.append(row)
            pair_rows.append(row)
        margins, aggregate = summarize_pairs(scored)
        trajectory.append(
            {
                "prompt": prompt_name,
                "lens_index": layer,
                "location": lens_location(layer),
                "aggregate_margin": aggregate,
                "pair1_margin": margins[1],
                "pair2_margin": margins[2],
                "pair3_margin": margins[3],
                "mean_kl_to_checkpoint_output": sum(kls) / len(kls),
            }
        )

    scored = []
    for row_index, entry in enumerate(entries):
        start = entry["start"]
        stop = start + len(entry["targets"])
        logits = output.logits[row_index, start - 1 : stop - 1, :]
        value = mean_target_logp(logits, entry["targets"])
        row = {
            "prompt": prompt_name,
            "lens_index": "model",
            "location": "model_output",
            "pair": entry["pair"],
            "kind": entry["kind"],
            "mean_logp": value,
            "boundary_agrees": entry["boundary_agrees"],
        }
        scored.append(row)
        pair_rows.append(row)
    margins, aggregate = summarize_pairs(scored)
    trajectory.append(
        {
            "prompt": prompt_name,
            "lens_index": "model",
            "location": "model_output",
            "aggregate_margin": aggregate,
            "pair1_margin": margins[1],
            "pair2_margin": margins[2],
            "pair3_margin": margins[3],
            "mean_kl_to_checkpoint_output": 0.0,
        }
    )
    del output, hidden_states
    gc.collect()
    return pair_rows, trajectory


def default_output():
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    return (
        HERE
        / "runs"
        / "pythia-410m-bme-tuned-lens"
        / f"run-{run_id}-attempt-{attempt}"
    )


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def run(args):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    output_dir = args.output or default_output()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=FINAL_REVISION)
    api = HfApi()
    resolved = {
        revision: api.model_info(MODEL, revision=revision).sha
        for revision in (*CHECKPOINTS, FINAL_REVISION)
    }

    frozen_lens = None
    frozen_metadata = None
    if args.mode == "frozen-final":
        frozen_lens, frozen_metadata = load_public_frozen_lens()

    all_pairs = []
    all_trajectory = []
    manifest_rows = []
    for checkpoint in CHECKPOINTS:
        print(f"loading {checkpoint}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=checkpoint, torch_dtype=torch.float32
        )
        model.eval()
        if args.mode == "frozen-final":
            lens = frozen_lens
            lens_metadata = dict(frozen_metadata)
        else:
            lens, lens_metadata = load_checkpoint_lens(
                model, Path(args.lens_root) / checkpoint, checkpoint
            )

        for prompt_name, prompt in PROMPTS.items():
            print(f"{checkpoint} {prompt_name}", flush=True)
            pair_rows, trajectory = score_prompt(
                model, lens, tokenizer, prompt_name, prompt
            )
            for row in pair_rows:
                row["checkpoint"] = checkpoint
                row["mode"] = args.mode
            for row in trajectory:
                row["checkpoint"] = checkpoint
                row["mode"] = args.mode
            all_pairs.extend(pair_rows)
            all_trajectory.extend(trajectory)

        manifest_rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha": resolved[checkpoint],
                "mode": args.mode,
                **lens_metadata,
            }
        )
        if args.mode == "checkpoint-specific":
            del lens
        del model
        gc.collect()

    tsv(
        output_dir / "pair_scores.tsv",
        [
            "mode", "checkpoint", "prompt", "lens_index", "location",
            "pair", "kind", "mean_logp", "boundary_agrees"
        ],
        all_pairs,
    )
    tsv(
        output_dir / "trajectory.tsv",
        [
            "mode", "checkpoint", "prompt", "lens_index", "location",
            "aggregate_margin", "pair1_margin", "pair2_margin", "pair3_margin",
            "mean_kl_to_checkpoint_output"
        ],
        all_trajectory,
    )
    tsv(
        output_dir / "manifest.tsv",
        [
            "checkpoint", "checkpoint_sha", "mode", "lens_resource",
            "lens_params_sha256", "lens_base_model",
            "lens_base_model_revision", "lens_unembed_hash",
            "decoder_unembed_hash", "decoder_revision"
        ],
        manifest_rows,
    )

    original = {
        checkpoint: [
            row for row in all_trajectory
            if row["checkpoint"] == checkpoint
            and row["prompt"] == "bme_original"
        ]
        for checkpoint in CHECKPOINTS
    }
    lines = [
        "# Pythia-410M BME Tuned Lens trajectory",
        "",
        f"- mode: {args.mode}",
        f"- model: {MODEL}",
        (
            "- score: original three-pair mean of school-pipeline minus "
            "job-conversion-check mean token log probability"
        ),
        (
            "- prompts: original BME, held-out BME, and unrelated "
            "school-question control"
        ),
        "",
        "## Evidence boundary",
        "",
    ]
    if args.mode == "frozen-final":
        lines += [
            (
                "This run applies the public final-model Pythia-410M Tuned Lens "
                "as one fixed decoder to both early checkpoints. It is a "
                "cross-checkpoint transfer diagnostic, not a checkpoint-calibrated "
                "Tuned Lens result. A sign change at a lens layer can show changing "
                "alignment to the final model decoder, but cannot by itself establish "
                "where either early checkpoint own prediction became decodable."
            ),
            "",
        ]
    else:
        lines += [
            (
                "Each checkpoint uses a separately trained Tuned Lens whose saved "
                "model revision and unembedding hash were required to match that "
                "checkpoint. Layer comparisons are descriptive readouts, not causal "
                "interventions."
            ),
            "",
        ]

    lines += ["## Original BME trajectory summary", ""]
    for checkpoint in CHECKPOINTS:
        rows = original[checkpoint]
        model_row = rows[-1]
        lens_rows = rows[:-1]
        first_pos = first_positive(lens_rows)
        positive_final = model_row["aggregate_margin"] > 0
        sustained = first_sustained_sign(lens_rows, positive_final)
        lines += [
            f"### {checkpoint}",
            "",
            f"- model-output margin: {model_row['aggregate_margin']:+.6f}",
            (
                "- first positive lens location: "
                f"{first_pos if first_pos is not None else 'none'}"
            ),
            (
                "- first residual location after which the lens sign stays equal to "
                "the model-output sign: "
                f"{sustained if sustained is not None else 'none'}"
            ),
            "",
            "| lens index | residual location | aggregate | pair 1 | pair 2 | pair 3 | mean KL |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {lens_index} | {location} | {aggregate_margin:+.6f} | "
                "{pair1_margin:+.6f} | {pair2_margin:+.6f} | "
                "{pair3_margin:+.6f} | {mean_kl_to_checkpoint_output:.6f} |".format(**row)
            )
        lines.append("")

    lines += [
        "## Provenance",
        "",
        f"- generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- Transformers: {transformers.__version__}",
        f"- git commit: {git_value('rev-parse', 'HEAD')}",
        f"- git branch: {git_value('rev-parse', '--abbrev-ref', 'HEAD')}",
        "",
        (
            "Raw values: trajectory.tsv and pair_scores.tsv; exact model and "
            "lens identities: manifest.tsv."
        ),
    ]
    atomic_write(output_dir / "analysis.md", "\n".join(lines) + "\n")
    print(output_dir, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("frozen-final", "checkpoint-specific"),
        required=True,
    )
    parser.add_argument(
        "--lens-root",
        type=Path,
        help="directory containing step512/ and step1000/ lens artifacts",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "checkpoint-specific" and args.lens_root is None:
        parser.error("--lens-root is required for checkpoint-specific mode")
    return args


if __name__ == "__main__":
    run(parse_args())
