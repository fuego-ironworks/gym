#!/usr/bin/env python3
"""Compare adjacent MLP and attention outputs at the last prompt token."""

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

MLP_LAYERS = tuple(range(8)) + (12, 23)
ATTENTION_LAYERS = tuple(range(6)) + (8, 23)


@contextlib.contextmanager
def zero_last_prompt(model, kind, layer, index):
    def hook(_module, _inputs, result):
        tensor = result[0] if isinstance(result, tuple) else result
        altered = tensor.clone()
        altered[:, index:index + 1] = 0
        return (altered,) + result[1:] if isinstance(result, tuple) else altered

    handle = original.component(model, layer, kind).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def score(model, tokenizer, prompt, continuation, kind=None, layer=None):
    ids, start, boundary_agrees = original.encode(tokenizer, prompt, continuation)
    context = (zero_last_prompt(model, kind, layer, start - 1)
               if kind is not None else contextlib.nullcontext())
    with torch.inference_mode(), context:
        logits = model(ids, use_cache=False).logits[0, start - 1:-1, :].float()
        targets = ids[0, start:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).sum().item()
    return logp / targets.numel(), boundary_agrees


def pairs(model, tokenizer, prompt, kind=None, layer=None):
    out = []
    for number, (school, check) in enumerate(original.PAIRS, 1):
        s, s_agrees = score(model, tokenizer, prompt, school, kind, layer)
        c, c_agrees = score(model, tokenizer, prompt, check, kind, layer)
        out.append((number, s, c, s - c, s_agrees and c_agrees))
    return out


def margin(values):
    return sum(v[3] for v in values) / len(values)


def neutral(model, tokenizer, kind=None, layer=None):
    return sum(score(model, tokenizer, p, c, kind, layer)[0]
               for p, c in original.NEUTRAL) / len(original.NEUTRAL)


def main():
    torch.set_num_threads(2)
    run_key = ("run-" + os.environ["GITHUB_RUN_ID"] + "-attempt-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
               if "GITHUB_RUN_ID" in os.environ else "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = original.HERE / "runs" / "pythia-410m-bme-last-prompt-scan" / run_key
    tokenizer = AutoTokenizer.from_pretrained(original.MODEL, revision="step143000")
    api = HfApi()
    shas = {s: api.model_info(original.MODEL, revision=s).sha for s in (*original.CHECKPOINTS, "step143000")}
    rows, pair_rows = [], []
    for checkpoint in original.CHECKPOINTS:
        model = original.load_model(checkpoint)
        base = {name: pairs(model, tokenizer, original.PROMPTS[name])
                for name in ("bme_original", "bme_heldout")}
        neutral_base = neutral(model, tokenizer)
        for kind, layers in (("mlp", MLP_LAYERS), ("attention", ATTENTION_LAYERS)):
            for layer in layers:
                neutral_after = neutral(model, tokenizer, kind, layer)
                for name, baseline in base.items():
                    changed = pairs(model, tokenizer, original.PROMPTS[name], kind, layer)
                    rows.append(dict(checkpoint=checkpoint, component=kind, layer=layer,
                                     prompt=name, baseline=f"{margin(baseline):+.8f}",
                                     margin=f"{margin(changed):+.8f}",
                                     delta=f"{margin(changed) - margin(baseline):+.8f}",
                                     neutral_delta=f"{neutral_after - neutral_base:+.8f}"))
                    for number, school, check, pair_margin, agrees in changed:
                        pair_rows.append(dict(checkpoint=checkpoint, component=kind, layer=layer,
                                              prompt=name, pair=number, school_mean=f"{school:.8f}",
                                              check_mean=f"{check:.8f}", pair_margin=f"{pair_margin:+.8f}",
                                              boundary_agrees=agrees))
                print(checkpoint, kind, layer, rows[-2]["delta"], flush=True)
        if checkpoint == "step1000":
            checked = next(float(row["margin"]) for row in rows if row["checkpoint"] == checkpoint
                           and row["component"] == "mlp" and row["layer"] == 0
                           and row["prompt"] == "bme_original")
            if abs(checked - (-0.00296271)) > 1e-5:
                raise AssertionError("last-prompt layer-0 result disagrees with position-scopes receipt")
        del model

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=original.HERE, text=True).strip()
    metadata = [("model", original.MODEL), ("source_commit", source_commit),
                *(("checkpoint_" + ref, sha) for ref, sha in shas.items()),
                ("tokenizer_revision", "step143000"), ("torch", torch.__version__),
                ("transformers", transformers.__version__), ("python", platform.python_version()),
                ("device", "cpu"), ("dtype", "float32"), ("threads", torch.get_num_threads()),
                ("script_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())]
    original.tsv(output / "manifest.tsv", ("key", "value"), [dict(key=k, value=v) for k, v in metadata])
    original.tsv(output / "scan.tsv", ("checkpoint", "component", "layer", "prompt",
                                        "baseline", "margin", "delta", "neutral_delta"), rows)
    original.tsv(output / "pairs.tsv", ("checkpoint", "component", "layer", "prompt", "pair",
                                         "school_mean", "check_mean", "pair_margin", "boundary_agrees"), pair_rows)
    original.atomic_write(output / "README.md", """# Final-prompt-token neighborhood scan

Only the output of the named component at the **last prompt token** is zeroed;
all other token positions and components retain their original outputs. The
original three BME continuation pairs are scored for both the canonical and
held-out BME student at step512 and step1000. Three short neutral continuations
measure collateral language-model loss under the identical intervention.

All MLP layers 0–7 plus 12 and 23 are tested; attention layers 0–5 plus 8 and
23 provide neighboring and spaced comparisons. This is a bounded component
scan, not a claim that untested components are irrelevant. The layer-0
step1000 canonical score is checked against the independent token-scope run.
`scan.tsv` and `pairs.tsv` retain aggregate and candidate-level numbers.
Zeroing a component is causal for this likelihood score; it does not identify
a unique semantic representation or a training origin.
""")


if __name__ == "__main__":
    main()
