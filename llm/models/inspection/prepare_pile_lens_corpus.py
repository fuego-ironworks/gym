#!/usr/bin/env python3
"""Build a bounded deterministic Pile-validation corpus for lens calibration.

The BME evaluation wording is explicitly excluded. The corpus itself stays out
of Git; the manifest records the exact source revision and corpus hash.
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "EleutherAI/pile_val_test"
SPLIT = "validation"
TOKENIZER_MODEL = "EleutherAI/pythia-410m-deduped"
TOKENIZER_REVISION = "step143000"
EXCLUDE = (
    "biomedical engineering",
    "what schools should i be looking at",
    "biomedical engineering job pipeline",
    "biomedical engineering job outcomes",
    "checking whether biomedical engineering",
)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=65536)
    parser.add_argument("--max-document-tokens", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, revision=TOKENIZER_REVISION
    )
    dataset = load_dataset(
        DATASET,
        split=SPLIT,
        revision=args.revision,
        streaming=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    skipped = 0
    token_count = 0

    with args.output.open("w", encoding="utf-8") as out:
        for row in dataset:
            text = row.get("text") or ""
            lower = text.lower()
            if not text.strip() or any(phrase in lower for phrase in EXCLUDE):
                skipped += 1
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                skipped += 1
                continue
            if len(ids) > args.max_document_tokens:
                text = tokenizer.decode(
                    ids[: args.max_document_tokens],
                    clean_up_tokenization_spaces=False,
                )
                ids = tokenizer.encode(text, add_special_tokens=False)

            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            accepted += 1
            token_count += len(ids)
            if token_count >= args.target_tokens:
                break

    if token_count < args.target_tokens:
        raise RuntimeError(
            f"source exhausted at {token_count} tokens; "
            f"needed {args.target_tokens}"
        )

    corpus_hash = sha256_file(args.output)
    manifest = [
        "field\tvalue",
        f"source_dataset\t{DATASET}",
        f"source_revision\t{args.revision}",
        f"source_split\t{SPLIT}",
        f"tokenizer_model\t{TOKENIZER_MODEL}",
        f"tokenizer_revision\t{TOKENIZER_REVISION}",
        f"target_tokens\t{args.target_tokens}",
        f"collected_tokens\t{token_count}",
        f"accepted_documents\t{accepted}",
        f"skipped_documents\t{skipped}",
        f"max_document_tokens\t{args.max_document_tokens}",
        f"corpus_sha256\t{corpus_hash}",
        "excluded_phrases\t" + " | ".join(EXCLUDE),
    ]
    atomic_write(args.manifest, "\n".join(manifest) + "\n")
    print(f"corpus={args.output}")
    print(f"tokens={token_count}")
    print(f"sha256={corpus_hash}")


if __name__ == "__main__":
    main()
