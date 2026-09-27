#!/usr/bin/env python3
"""Render literal A/B prompts for manual or hosted-model testing."""

from __future__ import annotations

import argparse
import json
import pathlib

BLACKBALL_URL = "https://github.com/bl4ckb4ll/blackball"


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=pathlib.Path, default=here / "questions.jsonl")
    parser.add_argument("--output", type=pathlib.Path, default=here / "rendered-prompts")
    args = parser.parse_args()

    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        qid = item["id"]
        question = item["question"]
        root = args.output / qid
        root.mkdir(parents=True, exist_ok=True)
        (root / "without-blackball.txt").write_text(question + "\n", encoding="utf-8")
        (root / "blackball-url.txt").write_text(
            BLACKBALL_URL + "\n\n" + question + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
