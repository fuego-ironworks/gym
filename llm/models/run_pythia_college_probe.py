#!/usr/bin/env python3
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m-deduped"
REVISION = "step143000"
PROMPTS = Path(__file__).with_name("probes") / "college.json"
OUTPUT = Path(__file__).with_name("runs") / "pythia-410m-step143000-college.md"
MAX_NEW_TOKENS = 128
SAMPLE_SEEDS = (17, 29, 43)
TEMPERATURE = 0.8
TOP_P = 0.95


def continuation(tokenizer, full_ids, prompt_length):
    return tokenizer.decode(full_ids[prompt_length:], skip_special_tokens=True)


def generate_greedy(model, tokenizer, input_ids, attention_mask):
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return continuation(tokenizer, output[0], input_ids.shape[1])


def generate_sample(model, tokenizer, input_ids, attention_mask, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return continuation(tokenizer, output[0], input_ids.shape[1])


def block(text):
    return "\n".join("    " + line for line in text.splitlines()) or "    <empty>"


def main():
    cases = json.loads(PROMPTS.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION)
    model.eval()

    lines = [
        "# Pythia 410M college probes",
        "",
        f"- model: `{MODEL}`",
        f"- revision: `{REVISION}`",
        f"- transformers: `{transformers.__version__}`",
        f"- torch: `{torch.__version__}`",
        f"- max new tokens: `{MAX_NEW_TOKENS}`",
        "- prompt treatment: raw causal continuation; no system message, chat template, instruction prefix, or few-shot examples",
        f"- sampled continuations: temperature `{TEMPERATURE}`, top-p `{TOP_P}`, seeds `{', '.join(map(str, SAMPLE_SEEDS))}`",
        "",
    ]

    for case in cases:
        prompt = case["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        lines += [
            f"## {case['id']}",
            "",
            "### Prompt",
            "",
            block(prompt),
            "",
            "### Greedy continuation",
            "",
            block(generate_greedy(model, tokenizer, input_ids, attention_mask)),
            "",
        ]

        for seed in SAMPLE_SEEDS:
            lines += [
                f"### Sample seed {seed}",
                "",
                block(generate_sample(model, tokenizer, input_ids, attention_mask, seed)),
                "",
            ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
