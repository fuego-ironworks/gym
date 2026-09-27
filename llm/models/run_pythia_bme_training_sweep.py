#!/usr/bin/env python3
import gc
import json
import math
import shutil
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m-deduped"
CHECKPOINTS = (
    "step0",
    "step1000",
    "step5000",
    "step20000",
    "step60000",
    "step100000",
    "step143000",
)
PROBE = Path(__file__).with_name("probes") / "bme-training.json"
OUTPUT = Path(__file__).with_name("runs") / "pythia-410m-bme-training-sweep.md"
MAX_NEW_TOKENS = 64


def block(text):
    return "\n".join("    " + line for line in text.splitlines()) or "    <empty>"


def score_continuation(model, tokenizer, prompt, continuation):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]

    prompt_len = prompt_ids.shape[1]
    if not torch.equal(full_ids[:, :prompt_len], prompt_ids):
        raise RuntimeError("continuation changed prompt tokenization at the boundary")

    with torch.inference_mode():
        logits = model(full_ids).logits[:, :-1, :]
        labels = full_ids[:, 1:]
        token_logp = torch.log_softmax(logits, dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

    first_target = prompt_len - 1
    candidate_logp = token_logp[:, first_target:]
    total = candidate_logp.sum().item()
    count = candidate_logp.numel()
    return {
        "tokens": count,
        "total_logp": total,
        "mean_logp": total / count,
    }


def generate_greedy(model, tokenizer, prompt):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def cache_dir():
    return Path.home() / ".cache" / "huggingface" / "hub" / "models--EleutherAI--pythia-410m-deduped"


def main():
    case = json.loads(PROBE.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision="step143000")

    rows = []
    generations = []

    for revision in CHECKPOINTS:
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision)
        model.eval()

        evidence = score_continuation(
            model, tokenizer, case["prompt"], case["evidence_first"]
        )
        promotion = score_continuation(
            model, tokenizer, case["prompt"], case["promotion_first"]
        )
        greedy = generate_greedy(model, tokenizer, case["prompt"])

        rows.append({
            "revision": revision,
            "evidence": evidence,
            "promotion": promotion,
            "delta_total": evidence["total_logp"] - promotion["total_logp"],
            "delta_mean": evidence["mean_logp"] - promotion["mean_logp"],
        })
        generations.append((revision, greedy))

        del model
        gc.collect()
        shutil.rmtree(cache_dir(), ignore_errors=True)

    lines = [
        "# Pythia 410M biomedical-engineering training sweep",
        "",
        f"- model: \x60{MODEL}\x60",
        f"- transformers: \x60{transformers.__version__}\x60",
        f"- torch: \x60{torch.__version__}\x60",
        "- prompt treatment: raw causal continuation; no system message, chat template, instruction prefix, or few-shot examples",
        "- checkpoints: " + ", ".join(f"\x60{x}\x60" for x in CHECKPOINTS),
        "- delta definition: evidence-first minus promotion-first; positive values favor the evidence-first continuation",
        "- both total sequence log probability and mean log probability per continuation token are reported",
        "",
        "## Prompt",
        "",
        block(case["prompt"]),
        "",
        "## Candidate continuations",
        "",
        "### Evidence-first",
        "",
        block(case["evidence_first"]),
        "",
        "### Promotion-first",
        "",
        block(case["promotion_first"]),
        "",
        "## Scores",
        "",
        "| checkpoint | evidence tokens | evidence total | evidence mean | promotion tokens | promotion total | promotion mean | delta total | delta mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        e = row["evidence"]
        p = row["promotion"]
        lines.append(
            f"| {row['revision']} | {e['tokens']} | {e['total_logp']:.4f} | {e['mean_logp']:.4f} | "
            f"{p['tokens']} | {p['total_logp']:.4f} | {p['mean_logp']:.4f} | "
            f"{row['delta_total']:.4f} | {row['delta_mean']:.4f} |"
        )

    lines += [
        "",
        "## Greedy continuations",
        "",
    ]

    for revision, greedy in generations:
        lines += [
            f"### {revision}",
            "",
            block(greedy),
            "",
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
