"""Run the frozen Blackball BME generation baseline or continuation diagnostic."""
import argparse
from importlib import metadata
import json
import hashlib
from pathlib import Path
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_pythia_bme_training_sweep import score_continuation
from run_pythia_14m_lora_inspectability import (
    REPO_ROOT, atomic_json, file_sha256, source_receipt,
)

MODEL = "EleutherAI/pythia-410m-deduped"
REVISION = "c66f7467608ffee8fca0d28cf1f46a7574b53cec"
MODEL_SHA256 = "e5130ca26aa649f69035269bfb00a25afc4704d11ff4385ae368e6eed4ad530a"
TOKENIZER_BLOBS = {
    "config.json": "0425fa136ba3f95d9428d832cfd7bfd82c78bf1f",
    "tokenizer.json": "f74dfbfab8f97770a87769c739fb080c21c8bacc",
    "tokenizer_config.json": "f1860edb10f80bcaf7b023fce47c68a23b724c23",
    "special_tokens_map.json": "0204ed10c186a4c7c68f55dff8f26087a45898d6",
}
CASES = REPO_ROOT / "cases/blackball-bias/BB-PYTHIA-01/bme-advice"
SEED = 29


def checked_model(directory):
    weights = file_sha256(directory / "model.safetensors")
    if weights != MODEL_SHA256:
        raise ValueError("model weights do not match the pinned checkpoint")
    for name, expected in TOKENIZER_BLOBS.items():
        data = (directory / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise ValueError(f"source blob mismatch for {name}")
    return {"model.safetensors": weights,
            **{name: file_sha256(directory / name) for name in TOKENIZER_BLOBS}}


def read_cases(name):
    cases = [json.loads(line) for line in (CASES / name).read_text().splitlines()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    return cases


def render(question, evidence, wrapper):
    text = f"Question: {question}\nAnswer:" if wrapper == "qa" else question
    return f"{evidence}\n\n{text}" if evidence else text


@torch.inference_mode()
def generate(model, tokenizer, cases, args, output):
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    evidence = (CASES / "evidence.txt").read_text().rstrip()
    all_rows = []
    for start in range(0, len(cases), args.batch):
        batch = cases[start:start + args.batch]
        for condition in args.conditions:
            context = evidence if condition == "evidence" else ""
            prompts = [render(case["question"], context, args.wrapper) for case in batch]
            enc = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
            prompt_lengths = enc.attention_mask.sum(dim=1)
            for decoding in args.decoding:
                t0 = time.perf_counter()
                torch.manual_seed(SEED)
                sampling = {"temperature": 0.7, "top_p": 0.9} if decoding == "sample" else {}
                ids = model.generate(
                    **enc, max_new_tokens=args.tokens, do_sample=decoding == "sample",
                    pad_token_id=tokenizer.eos_token_id, **sampling,
                )[:, enc.input_ids.shape[1]:].tolist()
                for case, text, length, tokens in zip(batch, prompts, prompt_lengths, ids):
                    eos = tokenizer.eos_token_id in tokens
                    if eos:
                        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    all_rows.append({
                        "id": case["id"], "condition": condition, "decoding": decoding,
                        "prompt": text, "prompt_tokens": length.item(), "generated_ids": tokens,
                        "stop": "eos" if eos else "length",
                        "output": tokenizer.decode(tokens, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False),
                    })
                atomic_json(output / "generations.json", {"rows": all_rows})
                print(f"{batch[0]['id']}–{batch[-1]['id']} {condition} {decoding} "
                      f"{time.perf_counter() - t0:.2f}s", flush=True)
    return len(all_rows)


def score(model, tokenizer, cases, output):
    rows = []
    for case in cases:
        for condition, prefix in (
            ("bare-answer", "Answer:"),
            ("closed-book", render(case["prompt"], "", "qa")),
            ("evidence", render(case["prompt"], case["evidence"], "qa")),
        ):
            choices = {key: score_continuation(model, tokenizer, prefix, " " + case[key])
                       for key in ("preferred", "disfavored")}
            rows.append({"id": case["id"], "group": case["group"], "condition": condition,
                         "prompt": prefix, "scores": choices})
    atomic_json(output / "continuations.json", {"rows": rows})
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("generate", "score"), default="generate")
    parser.add_argument("--wrapper", choices=("literal", "qa"), default="literal")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--decoding", nargs="+", choices=("greedy", "sample"),
                        default=["greedy", "sample"])
    parser.add_argument("--conditions", nargs="+", choices=("closed-book", "evidence"),
                        default=["closed-book", "evidence"])
    args = parser.parse_args()
    if min(args.tokens, args.threads, args.batch) < 1:
        parser.error("tokens, threads and batch must be positive")
    t0 = time.perf_counter()
    source = source_receipt()
    files = checked_model(args.model_dir)
    name = "prompts.jsonl" if args.mode == "generate" else "continuations.jsonl"
    cases = read_cases(name)
    if args.ids:
        if set(args.ids) - {row["id"] for row in cases}:
            parser.error("unknown case ID")
        cases = [row for row in cases if row["id"] in args.ids]
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32, use_safetensors=True
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "source": source, "model": MODEL, "revision": REVISION, "files": files,
        "inputs": {filename: file_sha256(CASES / filename) for filename in (name, "evidence.txt")},
        "ids": [case["id"] for case in cases], "mode": args.mode, "precision": "float32",
        "threads": args.threads, "batch": args.batch, "wrapper": args.wrapper,
        "decoding": args.decoding, "conditions": args.conditions,
        "max_new_tokens": args.tokens, "sampling_seed_per_batch": SEED,
        "temperature": 0.7, "top_p": 0.9,
        "setup_seconds": round(time.perf_counter() - t0, 3),
        "versions": {name: metadata.version(name) for name in ("torch", "transformers")},
    }
    atomic_json(args.output_dir / "plan.json", receipt)
    if args.mode == "generate":
        count = generate(model, tokenizer, cases, args, args.output_dir)
    else:
        count = score(model, tokenizer, cases, args.output_dir)
    receipt.update(rows=count, elapsed_seconds=round(time.perf_counter() - t0, 3),
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(args.output_dir / "receipt.json", receipt)


if __name__ == "__main__":
    main()
