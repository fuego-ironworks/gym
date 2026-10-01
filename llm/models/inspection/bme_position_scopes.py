#!/usr/bin/env python3
"""Test whether the undocumented old layer-0 claim depended on token position.

Companion to bme_causal_recheck.py. The original experiment's checkpoint,
prompt, candidate and separate-tokenization choices remain unchanged.
"""

import contextlib
import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoTokenizer

import bme_causal_recheck as original

SCOPES = ("prompt", "last_prompt", "continuation")
CONTEXTS = ("bme_original", "bme_heldout", "bme_job_question", "unrelated_school_question")


@contextlib.contextmanager
def scoped_hook(model, scope, start, mode, donor=None):
    def hook(_module, _inputs, output):
        current = output[0] if isinstance(output, tuple) else output
        replacement = torch.zeros_like(current) if mode == "zero" else donor
        if replacement is None or replacement.shape != current.shape:
            raise ValueError("source/destination activation shape mismatch")
        edited = current.clone()
        if scope == "prompt":
            edited[:, :start] = replacement[:, :start]
        elif scope == "last_prompt":
            edited[:, start - 1:start] = replacement[:, start - 1:start]
        elif scope == "continuation":
            edited[:, start:] = replacement[:, start:]
        else:
            raise ValueError(scope)
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    handle = original.component(model, 0, "mlp").register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def score(model, tokenizer, prompt, continuation, scope=None, mode=None, donor_model=None):
    ids, start, boundary_agrees = original.encode(tokenizer, prompt, continuation)
    donor = None
    if donor_model is not None:
        donor = original.score_one(donor_model, tokenizer, prompt, continuation,
                                   {"layer": 0, "kind": "mlp", "mode": "capture"})["capture"]
    with torch.inference_mode():
        with scoped_hook(model, scope, start, mode, donor) if mode else contextlib.nullcontext():
            logits = model(ids, use_cache=False).logits[0, start - 1:-1, :].float()
            targets = ids[0, start:]
            logp = torch.log_softmax(logits, -1).gather(-1, targets[:, None]).sum().item()
    return logp / targets.numel(), boundary_agrees


def scored_pairs(model, tokenizer, prompt, scope=None, mode=None, donor_model=None):
    out = []
    for n, (school, check) in enumerate(original.PAIRS, 1):
        a, agrees_a = score(model, tokenizer, prompt, school, scope, mode, donor_model)
        b, agrees_b = score(model, tokenizer, prompt, check, scope, mode, donor_model)
        out.append((n, a, b, a - b, agrees_a and agrees_b))
    return out


def margin(pairs):
    return sum(pair[3] for pair in pairs) / len(pairs)


def neutral(model, tokenizer, scope=None):
    return sum(score(model, tokenizer, p, c, scope, "zero" if scope else None)[0]
               for p, c in original.NEUTRAL) / len(original.NEUTRAL)


def main():
    torch.set_num_threads(2)
    run_key = ("run-" + os.environ["GITHUB_RUN_ID"] + "-attempt-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
               if "GITHUB_RUN_ID" in os.environ else "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = original.HERE / "runs" / "pythia-410m-bme-position-scopes" / run_key
    tokenizer = AutoTokenizer.from_pretrained(original.MODEL, revision="step143000")
    api = HfApi()
    shas = {s: api.model_info(original.MODEL, revision=s).sha for s in (*original.CHECKPOINTS, "step143000")}
    models = {s: original.load_model(s) for s in original.CHECKPOINTS}
    rows, baselines, neutral_baselines = [], {}, {}

    for checkpoint, model in models.items():
        neutral_baselines[checkpoint] = neutral(model, tokenizer)
        for name in CONTEXTS:
            pairs = scored_pairs(model, tokenizer, original.PROMPTS[name])
            baselines[checkpoint, name] = margin(pairs)
            for number, school, check, pair_margin, agrees in pairs:
                rows.append(dict(destination=checkpoint, source="", scope="none", action="baseline",
                                 prompt=name, pair=number, school_mean=f"{school:.8f}",
                                 check_mean=f"{check:.8f}", pair_margin=f"{pair_margin:+.8f}",
                                 baseline=f"{margin(pairs):+.8f}", margin=f"{margin(pairs):+.8f}",
                                 delta="", neutral_delta="", boundary_agrees=agrees))
        for scope in SCOPES:
            neutral_loss = neutral(model, tokenizer, scope) - neutral_baselines[checkpoint]
            for name in CONTEXTS:
                pairs = scored_pairs(model, tokenizer, original.PROMPTS[name], scope, "zero")
                for number, school, check, pair_margin, agrees in pairs:
                    rows.append(dict(destination=checkpoint, source="", scope=scope, action="zero",
                                     prompt=name, pair=number, school_mean=f"{school:.8f}",
                                     check_mean=f"{check:.8f}", pair_margin=f"{pair_margin:+.8f}",
                                     baseline=f"{baselines[checkpoint, name]:+.8f}",
                                     margin=f"{margin(pairs):+.8f}",
                                     delta=f"{margin(pairs) - baselines[checkpoint, name]:+.8f}",
                                     neutral_delta=f"{neutral_loss:+.8f}", boundary_agrees=agrees))
                print(checkpoint, scope, name, "zero", f"{margin(pairs):+.6f}", flush=True)

    for dest_name, model in models.items():
        source_name = next(s for s in original.CHECKPOINTS if s != dest_name)
        for scope in SCOPES:
            for name in CONTEXTS:
                prompt = original.PROMPTS[name]
                pairs = scored_pairs(model, tokenizer, prompt, scope, "replace", models[source_name])
                for number, school, check, pair_margin, agrees in pairs:
                    rows.append(dict(destination=dest_name, source=source_name, scope=scope,
                                     action="replace", prompt=name, pair=number,
                                     school_mean=f"{school:.8f}", check_mean=f"{check:.8f}",
                                     pair_margin=f"{pair_margin:+.8f}",
                                     baseline=f"{baselines[dest_name, name]:+.8f}",
                                     margin=f"{margin(pairs):+.8f}",
                                     delta=f"{margin(pairs) - baselines[dest_name, name]:+.8f}",
                                     neutral_delta="", boundary_agrees=agrees))
                print(dest_name, source_name, scope, name, "replace", f"{margin(pairs):+.6f}", flush=True)
    for scope in SCOPES:
        identity = scored_pairs(models["step1000"], tokenizer, original.PROMPT,
                                scope, "replace", models["step1000"])
        if abs(margin(identity) - baselines["step1000", "bme_original"]) > 1e-5:
            raise AssertionError(f"self patch changed the margin at {scope}")

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=original.HERE, text=True).strip()
    metadata = [
        ("model", original.MODEL), ("source_commit", source_commit),
        *(("checkpoint_" + ref, sha) for ref, sha in shas.items()),
        ("tokenizer_revision", "step143000"), ("torch", torch.__version__),
        ("transformers", transformers.__version__), ("python", platform.python_version()),
        ("device", "cpu"), ("dtype", "float32"), ("threads", torch.get_num_threads()),
        ("scoring", original.SCORING),
        ("script_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    ]
    original.tsv(output / "manifest.tsv", ("key", "value"),
                 [dict(key=k, value=v) for k, v in metadata])
    original.tsv(output / "pairs.tsv", ("destination", "source", "scope", "action", "prompt", "pair",
                                         "school_mean", "check_mean", "pair_margin", "baseline", "margin",
                                         "delta", "neutral_delta", "boundary_agrees"), rows)
    original.atomic_write(output / "README.md", """# Layer-0 token-position recheck

The earlier exploratory layer-0 claim has no recorded hook or token-position
definition. This run tests three distinct intervention scopes on the exact
Pythia-410M step512/step1000 three-pair likelihood contrast:

- `prompt`: every prompt token's layer-0 MLP output;
- `last_prompt`: the last prompt token's layer-0 MLP output;
- `continuation`: only the teacher-forced continuation positions.

`zero` deletes the chosen output region. `replace` copies the same region from
the other checkpoint, run on the identical token sequence. The recipient's
other positions remain intact. `pairs.tsv` records all per-candidate scores,
pair and aggregate margins, neutral score changes for ablation, and an exact
token-boundary check. The self-replacement control must leave the baseline
unchanged under all three scopes.

The previous all-position scan remains a separate receipt. None of these
whole-vector substitutions identifies a unique feature or a training cause.
If one scope reverses the score, that observation is specific to the stated
tokens, continuations, checkpoint direction, and model output hook.
""")


if __name__ == "__main__":
    main()
