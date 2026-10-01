#!/usr/bin/env python3
"""Recheck the Pythia-410M step512/1000 BME contrast and component interventions.

Uses the exact three pairs and the original separate prompt/continuation encoding
from run_pythia_bme_onset.py. This is a base-model likelihood experiment, not an
assessment of the quality of advice or a semantic feature-discovery procedure.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_bme_onset", HERE / "run_pythia_bme_onset.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

MODEL = original.MODEL
PROMPT = original.PROMPT
PAIRS = original.PAIRS
CHECKPOINTS = ("step512", "step1000")
SCORING = "separately encode prompt and continuation; mean target logp per continuation token"

# The same six scored continuations are applied to every semantic prompt. A
# margin on a mismatched question is a collateral measure, not a claim that
# the question itself invites college advice.
PROMPTS = {
    "bme_original": PROMPT,
    "bme_heldout": (
        "I'm a senior in high school and taking BC calculus. I have a 3.8 GPA "
        "and a 760 on the math sat. I think I'm interested in biomedical "
        "engineering as a college major because I want to help people. "
        "I swim and play basketball. What schools should I be looking at for biomedical engineering?"
    ),
    "grades_without_college": (
        "I'm a junior in high school and will be taking AB calculus next year. "
        "I have a 3.7 GPA and a 780 on the math sat. I want to help people "
        "and run track and play soccer. What kinds of work should I look into?"
    ),
    "bme_job_question": (
        "I'm a junior in high school and will be taking AB calculus next year. "
        "I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in "
        "biomedical engineering as a job because I want to help people. "
        "I run track and I play soccer. What jobs should I be looking at in biomedical engineering?"
    ),
    "help_people_other_major": (
        "I'm a junior in high school and will be taking AB calculus next year. "
        "I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in "
        "environmental engineering as a college major because I want to help people. "
        "I run track and I play soccer. What schools should I be looking at for environmental engineering?"
    ),
    "unrelated_school_question": (
        "I'm a junior in high school and will be taking AB calculus next year. "
        "I have a 3.7 GPA and a 780 on the math sat. I think I'm interested in "
        "English literature as a college major because I enjoy reading. "
        "I run track and I play soccer. What schools should I be looking at for English literature?"
    ),
}

NEUTRAL = (
    ("The library opened at nine on Tuesday.", " The first visitor returned a book."),
    ("A carpenter measured the wooden board twice.", " Then she cut it to length."),
    ("The rain ended before the afternoon walk.", " The streets dried in the sun."),
)


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def tsv(path, headings, rows):
    def safe(value):
        return str(value).replace("\t", "\\t").replace("\n", "\\n")

    atomic_write(path, "\t".join(headings) + "\n" + "\n".join(
        "\t".join(safe(row.get(column, "")) for column in headings) for row in rows
    ) + "\n")


def encode(tokenizer, prompt, continuation):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(continuation, add_special_tokens=False)
    complete = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    assert prefix and suffix
    return torch.tensor([prefix + suffix], dtype=torch.long), len(prefix), prefix + suffix == complete


def component(model, layer, kind):
    block = model.gpt_neox.layers[layer]
    return block.mlp if kind == "mlp" else block.attention


@contextlib.contextmanager
def intervention(model, layer=None, kind=None, mode=None, source=None):
    if mode is None:
        yield None
        return

    captured = []

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if mode == "capture":
            captured.append(tensor.detach().clone())
            return output
        if mode == "zero":
            replacement = torch.zeros_like(tensor)
        elif mode == "replace":
            if source is None or source.shape != tensor.shape:
                raise ValueError(f"patch shape mismatch: {getattr(source, 'shape', None)} != {tensor.shape}")
            replacement = source.to(tensor.device, dtype=tensor.dtype)
        else:
            raise ValueError(mode)
        return (replacement,) + output[1:] if isinstance(output, tuple) else replacement

    # All scored examples intervene on the entire teacher-forced sequence.
    handle = component(model, layer, kind).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def score_one(model, tokenizer, prompt, continuation, change=None):
    ids, start, boundary_agrees = encode(tokenizer, prompt, continuation)
    with torch.inference_mode():
        with intervention(model, **(change or {})) as captured:
            logits = model(ids, use_cache=False).logits[0, start - 1:-1, :].float()
            targets = ids[0, start:]
            logp = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).sum().item()
    return {"mean": logp / targets.numel(), "total": logp, "tokens": targets.numel(),
            "boundary_agrees": boundary_agrees, "capture": captured[0] if captured else None}


def score_pairs(model, tokenizer, prompt, change=None, source_model=None, layer=None, kind=None):
    scores = []
    for school, check in PAIRS:
        candidates = []
        for continuation in (school, check):
            if source_model is None:
                result = score_one(model, tokenizer, prompt, continuation, change)
            else:
                donor = score_one(source_model, tokenizer, prompt, continuation,
                                  {"layer": layer, "kind": kind, "mode": "capture"})
                result = score_one(model, tokenizer, prompt, continuation,
                                   {"layer": layer, "kind": kind, "mode": "replace",
                                    "source": donor["capture"]})
            candidates.append(result)
        scores.append(candidates)
    return scores


def margin(scores):
    return sum(school["mean"] - check["mean"] for school, check in scores) / len(scores)


def add_pairs(rows, checkpoint, prompt_name, action, scores, layer="", kind="", donor=""):
    for n, (school, check) in enumerate(scores, 1):
        rows.append({"checkpoint": checkpoint, "prompt": prompt_name, "action": action,
                     "layer": layer, "component": kind, "donor": donor, "pair": n,
                     "school_mean": f"{school['mean']:.8f}", "check_mean": f"{check['mean']:.8f}",
                     "margin": f"{school['mean'] - check['mean']:.8f}",
                     "school_tokens": school["tokens"], "check_tokens": check["tokens"],
                     "boundary_agrees": school["boundary_agrees"] and check["boundary_agrees"]})


def load_model(revision):
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float32)
    model.eval()
    if not hasattr(model, "gpt_neox") or len(model.gpt_neox.layers) != 24:
        raise RuntimeError("expected a 24-layer GPT-NeoX model; inspect architecture")
    return model


def neutral_mean(model, tokenizer, change=None):
    return sum(score_one(model, tokenizer, p, c, change)["mean"] for p, c in NEUTRAL) / len(NEUTRAL)


def run(output, quick=False):
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision="step143000")
    api = HfApi()
    resolved = {s: api.model_info(MODEL, revision=s).sha for s in (*CHECKPOINTS, "step143000")}
    pair_rows, scan_rows, control_rows, patch_rows = [], [], [], []
    base = {}
    neutral_base = {}
    # Every MLP is checked. Attentions near the first layer plus widely spaced
    # later layers distinguish a neighboring effect from a broad disruption.
    attention_layers = (0, 1, 2, 3, 4, 5, 8, 12, 18, 23)
    mlp_layers = (0, 1, 2, 3) if quick else range(24)
    control_layers = (0, 1, 2, 3, 8, 16, 23)
    control_prompts = ("bme_original", "bme_heldout", "bme_job_question",
                       "unrelated_school_question")

    for checkpoint in CHECKPOINTS:
        print("baseline and scan", checkpoint, flush=True)
        model = load_model(checkpoint)
        baseline = score_pairs(model, tokenizer, PROMPT)
        add_pairs(pair_rows, checkpoint, "bme_original", "baseline", baseline)
        base[checkpoint] = margin(baseline)
        neutral_base[checkpoint] = neutral_mean(model, tokenizer)
        for name, prompt in PROMPTS.items():
            result = baseline if name == "bme_original" else score_pairs(model, tokenizer, prompt)
            control_rows.append({"checkpoint": checkpoint, "prompt": name, "action": "baseline",
                                 "layer": "", "margin": f"{margin(result):.8f}", "delta": ""})
            if name != "bme_original":
                add_pairs(pair_rows, checkpoint, name, "baseline", result)
        for kind, layers in (("mlp", mlp_layers), ("attention", attention_layers if not quick else (0, 1, 2, 3))):
            for layer in layers:
                change = {"layer": layer, "kind": kind, "mode": "zero"}
                result = score_pairs(model, tokenizer, PROMPT, change)
                ablated = margin(result)
                neutral_ablated = neutral_mean(model, tokenizer, change)
                scan_rows.append({"checkpoint": checkpoint, "component": kind, "layer": layer,
                                  "baseline": f"{base[checkpoint]:.8f}", "ablated": f"{ablated:.8f}",
                                  "delta": f"{ablated - base[checkpoint]:+.8f}",
                                  "neutral_mean": f"{neutral_ablated:.8f}",
                                  "neutral_delta": f"{neutral_ablated - neutral_base[checkpoint]:+.8f}"})
                add_pairs(pair_rows, checkpoint, "bme_original", "zero", result, layer, kind)
                print(checkpoint, kind, layer, f"{ablated - base[checkpoint]:+.5f}", flush=True)
        for layer in control_layers:
            if layer not in mlp_layers:
                continue
            change = {"layer": layer, "kind": "mlp", "mode": "zero"}
            for name in control_prompts if layer != 0 else PROMPTS:
                prompt = PROMPTS[name]
                result = score_pairs(model, tokenizer, prompt, change)
                unchanged = next(float(r["margin"]) for r in control_rows
                                 if r["checkpoint"] == checkpoint and r["prompt"] == name and r["action"] == "baseline")
                control_rows.append({"checkpoint": checkpoint, "prompt": name, "action": "zero_mlp",
                                     "layer": layer, "margin": f"{margin(result):.8f}",
                                     "delta": f"{margin(result) - unchanged:+.8f}"})
                add_pairs(pair_rows, checkpoint, name, "zero", result, layer, "mlp")
        del model

    # Equal-architecture checkpoint patch: donor and destination receive the
    # identical token sequence. Source activations come from the donor's own
    # forward pass for that sequence. Both directions prevent a one-way rescue
    # from being mistaken for an isolated BME circuit.
    older, newer = (load_model(s) for s in CHECKPOINTS)
    for dest_name, dest, source_name, source in (("step512", older, "step1000", newer),
                                                  ("step1000", newer, "step512", older)):
        for layer in ((0, 1, 2, 3) if quick else control_layers):
            print("patch", dest_name, source_name, layer, flush=True)
            for name in control_prompts if layer == 0 else ("bme_original",):
                result = score_pairs(dest, tokenizer, PROMPTS[name], source_model=source, layer=layer, kind="mlp")
                unchanged = next(float(r["margin"]) for r in control_rows
                                 if r["checkpoint"] == dest_name and r["prompt"] == name and r["action"] == "baseline")
                patch_rows.append({"destination": dest_name, "source": source_name, "prompt": name,
                                   "component": "mlp", "layer": layer, "margin": f"{margin(result):.8f}",
                                   "baseline": f"{unchanged:.8f}", "delta": f"{margin(result) - unchanged:+.8f}"})
                add_pairs(pair_rows, dest_name, name, "replace", result, layer, "mlp", source_name)
    # Hook identity check: transferring an activation from the same model on
    # the same tokens must leave its score unchanged to numeric tolerance.
    self_patch = score_pairs(newer, tokenizer, PROMPT, source_model=newer, layer=0, kind="mlp")
    if abs(margin(self_patch) - base["step1000"]) > 1e-5:
        raise AssertionError("self replacement changed canonical score")

    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    model_shas = "\n".join(f"checkpoint_{s}\t{resolved[s]}" for s in resolved)
    manifest = (
        f"model\t{MODEL}\n{model_shas}\nsource_commit\t{code_commit}\n"
        f"tokenizer_revision\tstep143000\ntorch\t{torch.__version__}\n"
        f"transformers\t{transformers.__version__}\npython\t{platform.python_version()}\n"
        f"device\tcpu\ndtype\tfloat32\nthreads\t{torch.get_num_threads()}\n"
        f"platform\t{platform.platform()}\nmachine\t{platform.machine()}\n"
        f"scoring\t{SCORING}\nquick\t{quick}\n"
        f"script_sha256\t{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}\n"
        f"onset_source_sha256\t{hashlib.sha256((HERE / 'run_pythia_bme_onset.py').read_bytes()).hexdigest()}\n"
    )
    tsv(output / "manifest.tsv", ("key", "value"),
        [dict(zip(("key", "value"), line.split("\t", 1))) for line in manifest.splitlines()])
    tsv(output / "prompts.tsv", ("name", "prompt", "school", "check"),
        [{"name": name, "prompt": prompt, "school": school, "check": check}
         for name, prompt in PROMPTS.items() for school, check in PAIRS])
    tsv(output / "neutral.tsv", ("prompt", "continuation"),
        [{"prompt": p, "continuation": c} for p, c in NEUTRAL])
    tsv(output / "pairs.tsv", ("checkpoint", "prompt", "action", "component", "layer", "donor", "pair",
                               "school_mean", "check_mean", "margin", "school_tokens", "check_tokens", "boundary_agrees"), pair_rows)
    tsv(output / "scan.tsv", ("checkpoint", "component", "layer", "baseline", "ablated", "delta",
                              "neutral_mean", "neutral_delta"), scan_rows)
    tsv(output / "controls.tsv", ("checkpoint", "prompt", "action", "layer", "margin", "delta"), control_rows)
    tsv(output / "patches.tsv", ("destination", "source", "prompt", "component", "layer", "baseline", "margin", "delta"), patch_rows)
    readme = f"""# Pythia-410M BME causal recheck — machine receipt

Exact model revisions appear in `manifest.tsv`; the exact prompts, controls, and candidates
appear in `prompts.tsv` and `neutral.tsv`. Each per-pair likelihood is in `pairs.tsv`.
Positive margins favor the school continuations in this finite set. A positive
margin does not establish the model's preference about education or jobs generally.

Canonical margin: step512 {base['step512']:+.6f}; step1000 {base['step1000']:+.6f}.
Neutral baseline mean token log-probability: step512 {neutral_base['step512']:+.6f};
step1000 {neutral_base['step1000']:+.6f}. A self-patch at step1000 layer-0 MLP
left the canonical margin unchanged within 1e-5.

`scan.tsv` contains zero-ablation interventions and a neutral-language collateral
score; `controls.tsv` contains matched semantic/context controls; `patches.tsv`
contains bidirectional, same-token cross-checkpoint MLP-output replacement.
An ablation is a causal change to this score, but broad disruption of the model
can also change it. Cross-checkpoint activation replacement is another causal
intervention on the score; it does not identify a unique semantic feature, neuron,
or training example. Compare all layers, all three pairs, both directions,
and neutral/control collateral movement before calling layer 0 special.

This run contains no SAE, tuned lens, training-data inspection, or cross-model replication.
"""
    atomic_write(output / "README.md", readme)
    print(f"complete: {output}; step512={base['step512']:+.6f}; step1000={base['step1000']:+.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_key = ("run-" + os.environ["GITHUB_RUN_ID"] + "-attempt-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
               if "GITHUB_RUN_ID" in os.environ else "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output", type=Path,
                        default=HERE / "runs" / "pythia-410m-bme-causal-recheck" / run_key)
    parser.add_argument("--quick", action="store_true", help="smoke run; NOT a complete component scan")
    args = parser.parse_args()
    run(args.output, args.quick)
