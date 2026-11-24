"""Run the preregistered Blackball Weste LoRA comparison on CPU."""
from __future__ import annotations

import argparse
import json
import math
import random
from importlib import metadata
from pathlib import Path
import sys

from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_differential import attach_lora, frozen_parameter_hashes, named_hashes_digest
from run_pythia_14m_lora_inspectability import (
    REPO_ROOT, atomic_json, file_sha256, source_receipt, write_artifact_index,
)

MODEL = "EleutherAI/pythia-14m-deduped"
REVISION = "0412c27fd5dcade8d1ff02273769acfb0c71421b"
FILES = {
    "model.safetensors": "d03d421ac835a0bfb64b5922acd780b84fac65f5ae84c2db59afa77773112d83",
    "config.json": "5536386c1709fd29d9617fca802bb8f592071ac279f772b5c63a6be3db3488e2",
    "tokenizer.json": "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
    "tokenizer_config.json": "4d1c4fdcf128d28ffcc56e21a36f7002b5e754624c9c8cbd025df7fa58af8428",
    "generation_config.json": "4cc3ecaaf254b8eed6b256d588741da925bdc229060b9ba9e04f4ce1b3fe7d0b",
    "special_tokens_map.json": "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
}
CASES = REPO_ROOT / "cases/blackball-bias/BB-PYTHIA-01/weste"
TARGET = "gpt_neox.layers.5.mlp.dense_4h_to_h"
RANK = 8
LEARNING_RATE = 0.003
EPOCHS = 3
BATCH_SIZE = 4
SEEDS = (7, 19)
CONTEXTS = ("without_evidence", "with_evidence")
ORDERS = ((0, 1), (1, 0))


def read_cases(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty case file {path}")
    for row in rows:
        if len(set(row["options"])) != 2 or len(row["options"]) != 2:
            raise ValueError(f"expected two distinct options on {row['id']}")
        for context in CONTEXTS:
            if row[f"correct_{context}"] not in (0, 1) or not row[f"explanation_{context}"].strip():
                raise ValueError(f"invalid answer key on {row['id']}")
    return rows


def prompt(case, context, order):
    evidence = case["evidence"] if context == "with_evidence" else "No source excerpt supplied."
    options = case["options"]
    return (f"Evidence: {evidence}\nQuestion: {case['question']}\n"
            f"A. {options[order[0]]}\nB. {options[order[1]]}\nAnswer:")


def training_examples(tokenizer, cases):
    samples = []
    for case in cases:
        for context in CONTEXTS:
            for order in ORDERS:
                prefix = tokenizer.encode(prompt(case, context, order), add_special_tokens=False)
                key = order.index(case[f"correct_{context}"])
                tail = tokenizer.encode(
                    f" {'AB'[key]}\n{case[f'explanation_{context}']}", add_special_tokens=False,
                ) + [tokenizer.eos_token_id]
                if len(prefix) + len(tail) > 320:
                    raise ValueError(f"token budget exceeded for {case['id']}")
                samples.append((prefix + tail, [-100] * len(prefix) + tail))
    return samples


def fit(model, adapter, tokenizer, cases, seed):
    samples = training_examples(tokenizer, cases)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)
    sequence = list(range(len(samples)))
    rng = random.Random(seed)
    history = []
    for epoch in range(EPOCHS):
        rng.shuffle(sequence)
        for offset in range(0, len(sequence), BATCH_SIZE):
            batch = [samples[i] for i in sequence[offset:offset + BATCH_SIZE]]
            width = max(len(ids) for ids, _ in batch)
            inputs = torch.tensor([ids + [tokenizer.eos_token_id] * (width - len(ids))
                                   for ids, _ in batch])
            labels = torch.tensor([gold + [-100] * (width - len(gold)) for _, gold in batch])
            attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch])
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, labels=labels, attention_mask=attention, use_cache=False).loss
            if not math.isfinite(loss.item()):
                raise ValueError("nonfinite loss")
            loss.backward()
            optimizer.step()
            history.append({"epoch": epoch + 1, "step": len(history) + 1, "loss": loss.item()})
    return history


@torch.inference_mode()
def evaluate(model, tokenizer, cases):
    token_ids = [tokenizer.encode(key, add_special_tokens=False) for key in (" A", " B")]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("answer letter isn't a single token")
    rows = []
    for case in cases:
        for context in CONTEXTS:
            gold = case[f"correct_{context}"]
            for order in ORDERS:
                rendered = prompt(case, context, order)
                ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
                logits = model(**ids, use_cache=False).logits[0, -1]
                a, b = [logits[indices[0]].item() for indices in token_ids]
                key = "AB"[order.index(gold)]
                rows.append({"case": case["id"], "group": case["group"], "domain": case["domain"],
                             "pair": case["pair"], "context": context, "order": list(order),
                             "key": key, "logit_a": a, "logit_b": b,
                             "margin": (a - b) if key == "A" else (b - a)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = source_receipt()
    model_digests = {name: file_sha256(args.model_dir / name) for name in FILES}
    if model_digests != FILES:
        raise ValueError("model directory does not match the pinned Pythia files")
    train = read_cases(CASES / "train.jsonl")
    test = read_cases(CASES / "holdout.jsonl")
    if {row["group"] for row in train} & {row["group"] for row in test}:
        raise ValueError("a case group crosses the development and held-out files")
    if len({r["id"] for r in train + test}) != len(train) + len(test):
        raise ValueError("duplicate case id")
    input_digests = {name: file_sha256(CASES / name) for name in ("train.jsonl", "holdout.jsonl")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval()
    adapter = attach_lora(model, TARGET, rank=RANK, alpha=RANK, seed=SEEDS[0])
    frozen = named_hashes_digest(frozen_parameter_hashes(model))
    adapter.enabled = False
    baseline = evaluate(model, tokenizer, test)
    adapter.enabled = True
    if evaluate(model, tokenizer, test) != baseline:
        raise RuntimeError("a freshly attached adapter changed the baseline")
    atomic_json(args.output_dir / "baseline.json", {"rows": baseline})
    adapters = {}
    for seed in SEEDS:
        # Replace the wrapper with its original linear layer before attaching a fresh adapter.
        owner, _, leaf = TARGET.rpartition(".")
        setattr(model.get_submodule(owner), leaf, adapter.base)
        adapter = attach_lora(model, TARGET, rank=RANK, alpha=RANK, seed=seed)
        trace = fit(model, adapter, tokenizer, train, seed)
        if named_hashes_digest(frozen_parameter_hashes(model)) != frozen:
            raise RuntimeError("a frozen parameter changed")
        filename = args.output_dir / f"adapter-{seed}.safetensors"
        save_file(adapter.adapter_state_dict(), filename)
        if torch.count_nonzero(adapter.b).item() == 0:
            raise RuntimeError("training did not update the adapter")
        adapter.load_adapter_state_dict(load_file(filename))
        atomic_json(args.output_dir / f"seed-{seed}.json",
                    {"trace": trace, "rows": evaluate(model, tokenizer, test)})
        adapters[str(seed)] = file_sha256(filename)
        print(f"seed {seed}: {len(trace)} optimizer updates", flush=True)
    atomic_json(args.output_dir / "receipt.json", {
        "source": source, "model": MODEL, "revision": REVISION, "files": model_digests,
        "inputs": input_digests, "frozen_base": frozen, "seeds": SEEDS,
        "target": TARGET, "rank": RANK, "alpha": RANK, "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "versions": {name: metadata.version(name) for name in ("torch", "transformers", "safetensors")},
        "python": sys.version, "adapter_sha256": adapters,
    })
    write_artifact_index(args.output_dir)


if __name__ == "__main__":
    main()
