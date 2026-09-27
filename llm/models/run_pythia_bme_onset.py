#!/usr/bin/env python3
"""Measure when Pythia 410M begins preferring the biomedical-engineering
college/school pipeline over a job-conversion check.

This is a base-model learning-dynamics probe. It does not use the Pythia RAG bin
or preference pairs; those are interventions to compare later.
"""

import gc
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m-deduped"
OUTPUT = Path(__file__).with_name("runs") / "pythia-410m-bme-onset.md"

PROMPT = (
    "I'm a junior in high school and will be taking AB calculus next year. "
    "I have a 3.7 GPA and a 780 on the math sat. "
    "I think I'm interested in biomedical engineering as a college major "
    "because I want to help people. I run track and I play soccer. "
    "What schools should I be looking at for biomedical engineering?"
)

# These pairs deliberately keep the syntactic frame close while changing the
# next action: accept the prestige/major pipeline and shop for schools, or check
# whether the degree actually converts into the desired work.
PAIRS = [
    (
        "\n\nI would recommend looking at strong biomedical engineering programs first.",
        "\n\nI would recommend checking the biomedical engineering job pipeline first.",
    ),
    (
        "\n\nThe next step is to compare biomedical engineering schools.",
        "\n\nThe next step is to compare biomedical engineering job outcomes.",
    ),
    (
        "\n\nStart by finding colleges with strong biomedical engineering programs.",
        "\n\nStart by checking whether biomedical engineering leads to the work you want.",
    ),
]

EARLY_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_STEPS = EARLY_STEPS + [1000] + list(range(10000, 140001, 10000)) + [143000]

def requested_steps():
    raw = os.environ.get("PYTHIA_BME_STEPS", "").strip()
    if not raw:
        return DEFAULT_STEPS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def available_steps():
    refs = HfApi().list_repo_refs(MODEL)
    out = set()
    for ref in refs.branches:
        name = ref.name
        if name.startswith("step") and name[4:].isdigit():
            out.add(int(name[4:]))
    return out


def continuation_stats(model, tokenizer, prompt, continuation):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long)

    with torch.inference_mode():
        logits = model(ids).logits[0]
        logp = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    token_logps = []
    for absolute_pos in range(start, len(prompt_ids) + len(cont_ids)):
        target = ids[0, absolute_pos]
        token_logps.append(float(logp[absolute_pos - 1, target]))

    total = sum(token_logps)
    mean = total / len(token_logps)
    return total, mean, len(token_logps)


def score_step(step, tokenizer):
    cache_dir = tempfile.mkdtemp(prefix=f"pythia-{step}-")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=f"step{step}",
            cache_dir=cache_dir,
        )
        model.eval()

        pair_rows = []
        for school_text, check_text in PAIRS:
            school_total, school_mean, school_n = continuation_stats(
                model, tokenizer, PROMPT, school_text
            )
            check_total, check_mean, check_n = continuation_stats(
                model, tokenizer, PROMPT, check_text
            )
            pair_rows.append(
                {
                    "school_total": school_total,
                    "school_mean": school_mean,
                    "school_n": school_n,
                    "check_total": check_total,
                    "check_mean": check_mean,
                    "check_n": check_n,
                    "margin": school_mean - check_mean,
                }
            )

        margin = sum(row["margin"] for row in pair_rows) / len(pair_rows)
        return margin, pair_rows
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        shutil.rmtree(cache_dir, ignore_errors=True)


def first_positive_interval(rows):
    ordered = sorted(rows)
    previous = None
    for step in ordered:
        if rows[step][0] > 0:
            return previous, step
        previous = step
    return None


def refinement_steps(interval, available):
    if interval is None:
        return []
    lo, hi = interval
    if lo is None or hi <= 1000:
        return []
    start = ((lo // 1000) + 1) * 1000
    return [s for s in range(start, hi, 1000) if s in available]


def classify(margin):
    if margin > 0:
        return "school-pipeline"
    if margin < 0:
        return "job-conversion-check"
    return "tie"


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision="step143000")
    available = available_steps()

    planned = [s for s in requested_steps() if s in available]
    rows = {}

    for step in planned:
        margin, pairs = score_step(step, tokenizer)
        rows[step] = (margin, pairs)
        print(f"step{step}: margin={margin:+.6f}", flush=True)

    interval = first_positive_interval(rows)
    for step in refinement_steps(interval, available):
        if step not in rows:
            margin, pairs = score_step(step, tokenizer)
            rows[step] = (margin, pairs)
            print(f"step{step}: margin={margin:+.6f}", flush=True)

    ordered = sorted(rows)
    first_positive = next((s for s in ordered if rows[s][0] > 0), None)

    # "Stable" here means the first measured positive checkpoint after which
    # every later measured checkpoint remains positive. This is deliberately
    # stricter than a one-checkpoint blip.
    first_stable = None
    for i, step in enumerate(ordered):
        if rows[step][0] > 0 and all(rows[s][0] > 0 for s in ordered[i:]):
            first_stable = step
            break

    lines = [
        "# Pythia 410M biomedical-engineering onset sweep",
        "",
        f"- model: `{MODEL}`",
        "- experiment: raw base-model checkpoint sweep; no RAG, adapter, system message, chat template, or few-shot examples",
        "- score: mean over three paired continuations of (school-pipeline mean token log-probability - job-conversion-check mean token log-probability)",
        "- positive margin: model assigns higher normalized likelihood to accepting the BME-school pipeline",
        "- negative margin: model assigns higher normalized likelihood to checking degree-to-job conversion first",
        "- checkpoint schedule: early powers-of-two + coarse 10k sweep + 1k refinement around first coarse positive crossing",
        "",
        "## Prompt",
        "",
        "    " + PROMPT,
        "",
        "## Result",
        "",
    ]

    if first_positive is None:
        lines += [
            "- first measured positive checkpoint: **none**",
            "- first stable positive checkpoint: **none**",
        ]
    else:
        lines += [
            f"- first measured positive checkpoint: **step{first_positive}**",
            f"- first stable positive checkpoint: **{('step' + str(first_stable)) if first_stable is not None else 'none'}**",
        ]

    lines += [
        "",
        "The first-positive value is an onset estimate under this explicit contrast, not a claim that a single semantic feature suddenly appears at one exact optimizer step. A transient sign flip is kept distinct from a stable preference.",
        "",
        "## Checkpoint scores",
        "",
        "| checkpoint | margin | preferred continuation |",
        "| ---: | ---: | --- |",
    ]

    for step in ordered:
        margin = rows[step][0]
        lines.append(f"| step{step} | {margin:+.6f} | {classify(margin)} |")

    lines += [
        "",
        "## Pair-level scores",
        "",
        "Margins below are school-pipeline mean token log-probability minus job-conversion-check mean token log-probability.",
        "",
    ]

    for step in ordered:
        lines += [f"### step{step}", ""]
        for i, row in enumerate(rows[step][1], 1):
            lines.append(
                f"- pair {i}: {row['margin']:+.6f} "
                f"(school {row['school_mean']:+.6f}, check {row['check_mean']:+.6f})"
            )
        lines.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
