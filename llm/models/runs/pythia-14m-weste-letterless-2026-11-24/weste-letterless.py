"""Replay the Weste pilot with each candidate scored as a letterless answer."""

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--original-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gym / "llm/models"))

    import torch
    from safetensors.torch import load
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_differential import attach_lora, frozen_parameter_hashes, named_hashes_digest
    from run_pythia_14m_lora_inspectability import atomic_json, file_sha256, source_receipt
    from run_weste_hiring_pilot import CASES, CONTEXTS, FILES, MODEL, RANK, REVISION, SEEDS, TARGET
    from run_weste_hiring_pilot import evaluate, read_cases

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    archive_hash = file_sha256(args.original_run)
    if archive_hash != "f4e01e72d9e003e2a8e1a7b95113a5a01bcb9a284aa70dea4b2742eb0f787c42":
        raise ValueError("Original Weste archive SHA-256 differs")
    asset_hashes = {name: file_sha256(args.model_dir / name) for name in FILES}
    if asset_hashes != FILES:
        raise ValueError("Pythia model SHA-256 differs")
    train_path = CASES / "train.jsonl"
    holdout_path = CASES / "holdout.jsonl"
    holdout = read_cases(holdout_path)
    case_hashes = {path.name: file_sha256(path) for path in (train_path, holdout_path)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval()
    adapter = attach_lora(model, TARGET, rank=RANK, alpha=RANK, seed=SEEDS[0])
    base_hash = named_hashes_digest(frozen_parameter_hashes(model))
    variants = [("frozen-base", None, "baseline.json")]
    variants.extend((f"seed-{seed}", seed, f"seed-{seed}.json") for seed in SEEDS)

    # The bundle is small. Verify its index before using the embedded data.
    with zipfile.ZipFile(args.original_run) as archive:
        index = json.loads(archive.read("artifact-index.json"))
        from hashlib import sha256
        members = {}
        for item in index["files"]:
            payload = archive.read(item["path"])
            if len(payload) != item["size"] or sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Archived member differs: {item['path']}")
            members[item["path"]] = payload
    receipt = json.loads(members["receipt.json"])
    for key, value in {
        "model": MODEL, "revision": REVISION, "files": asset_hashes, "inputs": case_hashes,
        "frozen_base": base_hash, "seeds": list(SEEDS), "target": TARGET, "rank": RANK,
        "alpha": RANK,
    }.items():
        if receipt.get(key) != value:
            raise ValueError(f"The original run receipt differs on {key}")

    @torch.inference_mode()
    def score(prefix, candidate):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        joint_ids = tokenizer.encode(prefix + candidate, add_special_tokens=False)
        if joint_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Candidate changed prefix tokenization")
        tail = joint_ids[len(prefix_ids):]
        if not tail:
            raise ValueError("Candidate must include at least one token")
        input_ids = torch.tensor([joint_ids])
        logits = model(input_ids=input_ids, use_cache=False).logits[
            0, len(prefix_ids) - 1:len(joint_ids) - 1
        ]
        token_logp = logits.log_softmax(dim=-1).gather(1, torch.tensor(tail).reshape(-1, 1))
        scores = token_logp.reshape(-1).tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Nonfinite token probability")
        total = math.fsum(scores)
        return {"token_ids": tail, "token_logp": scores, "sum": total, "mean": total / len(tail)}

    rows = []
    control = {}
    for name, seed, run_file in variants:
        adapter.enabled = seed is not None
        if seed is not None:
            data = members[f"adapter-{seed}.safetensors"]
            if sha256(data).hexdigest() != receipt["adapter_sha256"][str(seed)]:
                raise ValueError(f"Unexpected adapter weights for {seed}")
            adapter.load_adapter_state_dict(load(data))

        # Restore the exact A/B scoring prompt and compare all raw logits with
        # the original run. Keep observed numerical spread explicit.
        old_rows = json.loads(members[run_file])["rows"]
        replay = evaluate(model, tokenizer, holdout)
        if len(replay) != len(old_rows):
            raise ValueError(f"Incomplete A/B replay for {name}")
        deviations = []
        ab = {}
        for raw, original in zip(replay, old_rows):
            for key in ("case", "group", "domain", "pair", "context", "order", "key"):
                if raw[key] != original[key]:
                    raise ValueError(f"Original trial identity differs: {key}")
            deviations.extend(abs(raw[key] - original[key]) for key in ("logit_a", "logit_b"))
            ab.setdefault((raw["case"], raw["context"]), []).append(raw["margin"])
        control[name] = {"ab_logit_max_absolute_error": max(deviations), "ab_trials": len(replay)}
        control[name]["replay_order_decision_changes"] = sum(
            (raw["margin"] > 0) != (original["margin"] > 0) for raw, original in zip(replay, old_rows)
        )
        print(name, control[name], flush=True)

        for case in holdout:
            candidates = [" " + answer for answer in case["options"]]
            lexical = [score("Answer:", answer) for answer in candidates]
            for context in CONTEXTS:
                evidence = case["evidence"] if context == "with_evidence" else "No source excerpt supplied."
                prefix = f"Evidence: {evidence}\nQuestion: {case['question']}\nAnswer:"
                conditioned = [score(prefix, answer) for answer in candidates]
                correct = case[f"correct_{context}"]
                wrong = 1 - correct
                if any(c["token_ids"] != p["token_ids"] for c, p in zip(conditioned, lexical)):
                    raise ValueError("Lexical control scored different continuation tokens")
                margins = {}
                for metric in ("mean", "sum"):
                    observed = conditioned[correct][metric] - conditioned[wrong][metric]
                    baseline = lexical[correct][metric] - lexical[wrong][metric]
                    margins[metric] = {"raw": observed, "lexical": baseline, "context_lift": observed - baseline}
                ab_margins = ab[case["id"], context]
                rows.append({
                    "model": name, "case": case["id"], "group": case["group"], "domain": case["domain"],
                    "condition": context, "correct_option_index": correct, "prefix": prefix,
                    "candidates": candidates, "conditioned": conditioned, "lexical": lexical,
                    "margins": margins, "original_ab_replay_margins": ab_margins,
                })
        if named_hashes_digest(frozen_parameter_hashes(model)) != base_hash:
            raise ValueError("Frozen weights changed")

    summaries = []
    for name, _, _ in variants:
        for context in CONTEXTS:
            selected = [row for row in rows if row["model"] == name and row["condition"] == context]
            summary = {"model": name, "condition": context, "cases": len(selected)}
            summary["ab_correct_both_orders"] = sum(min(row["original_ab_replay_margins"]) > 0 for row in selected)
            for metric in ("mean", "sum"):
                for mode in ("raw", "lexical", "context_lift"):
                    numbers = [row["margins"][metric][mode] for row in selected]
                    suffix = f"{metric}_{mode}"
                    summary[f"correct_{suffix}"] = sum(number > 0 for number in numbers)
                    summary[f"ties_{suffix}"] = sum(number == 0 for number in numbers)
                    summary[f"average_margin_{suffix}"] = math.fsum(numbers) / len(numbers)
            summaries.append(summary)

    atomic_json(args.output, {
        "source": source_receipt(args.gym),
        "script_sha256": file_sha256(Path(__file__)),
        "assets": asset_hashes,
        "original_run_archive_sha256": archive_hash,
        "release": {"repo": MODEL, "revision": REVISION}, "runtime": {
            "threads": 2, "device": "cpu", "dtype": "float32", "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "python": sys.version,
        },
        "frozen_base": base_hash, "original_run": receipt, "controls": control,
        "lexical_prefix": "Answer:", "candidate_tokens_include_eos": False,
        "summaries": summaries, "rows": rows,
    })


if __name__ == "__main__":
    main()

