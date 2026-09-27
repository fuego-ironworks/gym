#!/usr/bin/env python3
"""Run paired college-question samples against a local Ollama model.

Primary conditions:
  without-blackball: literal question
  blackball-url:     Blackball URL + blank line + literal question

A local model normally cannot retrieve the URL. This runner therefore provides
reproducible local samples and a URL-cue negative control. It does not pretend
that the model read the repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys
import time
import urllib.request

BLACKBALL_URL = "https://github.com/bl4ckb4ll/blackball"
CONDITIONS = ("without-blackball", "blackball-url")


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen3:4b")
    parser.add_argument("--samples", type=int, default=5, help="paired samples per question")
    parser.add_argument("--questions", type=pathlib.Path, default=here / "questions.jsonl")
    parser.add_argument("--results", type=pathlib.Path, default=here / "results")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1000, help="base seed; sample N uses seed+N")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--blackball-ref",
        default=None,
        help="metadata only; commit/ref if a separate retrieval layer actually used Blackball",
    )
    return parser.parse_args()


def load_questions(path: pathlib.Path) -> list[dict]:
    questions: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item.get("id")
            question = item.get("question")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{path}:{line_number}: missing string id")
            if qid in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {qid}")
            if not isinstance(question, str) or not question:
                raise ValueError(f"{path}:{line_number}: missing string question")
            seen.add(qid)
            questions.append(item)
    return questions


def effective_prompt(question: str, condition: str) -> str:
    if condition == "without-blackball":
        return question
    if condition == "blackball-url":
        return f"{BLACKBALL_URL}\n\n{question}"
    raise ValueError(condition)


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value or "unnamed"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def call_ollama(host: str, payload: dict) -> dict:
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_metadata(question_item: dict) -> dict:
    return {
        "question_id": question_item["id"],
        "question": question_item["question"],
        "family": question_item.get("family"),
        "signal": question_item.get("signal"),
        "category": question_item.get("category"),
    }


def run_one(
    *,
    args: argparse.Namespace,
    question_item: dict,
    condition: str,
    sample_number: int,
    pair_seed: int,
    run_root: pathlib.Path,
) -> None:
    question = question_item["question"]
    prompt = effective_prompt(question, condition)
    condition_root = run_root / condition
    output_path = condition_root / f"sample-{sample_number:04d}.json"

    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
            "seed": pair_seed,
        },
    }

    for attempt in range(1, args.retries + 2):
        started_at = utc_now()
        try:
            backend = call_ollama(args.host, payload)
        except Exception as exc:  # preserve failures instead of making them disappear
            failure_path = (
                condition_root
                / "failures"
                / f"sample-{sample_number:04d}-attempt-{attempt:02d}.json"
            )
            write_json(
                failure_path,
                {
                    **question_metadata(question_item),
                    "condition": condition,
                    "effective_prompt": prompt,
                    "model": args.model,
                    "sample": sample_number,
                    "pair_seed": pair_seed,
                    "attempt": attempt,
                    "started_at": started_at,
                    "failed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if attempt <= args.retries:
                time.sleep(1)
                continue
            print(
                f"FAILED {question_item['id']} {condition} sample {sample_number}",
                file=sys.stderr,
            )
            return

        write_json(
            output_path,
            {
                **question_metadata(question_item),
                "condition": condition,
                "effective_prompt": prompt,
                "blackball_url": BLACKBALL_URL if condition == "blackball-url" else None,
                "blackball_ref": args.blackball_ref,
                "model": args.model,
                "sample": sample_number,
                "pair_seed": pair_seed,
                "sampling": payload["options"],
                "started_at": started_at,
                "finished_at": utc_now(),
                "response": backend.get("response", ""),
                "thinking": backend.get("thinking"),
                "raw_backend": backend,
            },
        )
        print(f"OK {question_item['id']} {condition} sample {sample_number}")
        return


def main() -> int:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")

    questions = load_questions(args.questions)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = safe_component(args.model)

    for question_item in questions:
        run_root = args.results / question_item["id"] / model_dir / run_id
        write_json(
            run_root / "run.json",
            {
                "run_id": run_id,
                **question_metadata(question_item),
                "model": args.model,
                "conditions": list(CONDITIONS),
                "samples": args.samples,
                "base_seed": args.seed,
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_predict": args.num_predict,
                },
                "blackball_url": BLACKBALL_URL,
                "blackball_ref": args.blackball_ref,
                "runner": "run_ollama.py",
                "created_at": utc_now(),
            },
        )

        for sample_number in range(1, args.samples + 1):
            pair_seed = args.seed + sample_number - 1
            order = CONDITIONS if sample_number % 2 else tuple(reversed(CONDITIONS))
            for condition in order:
                run_one(
                    args=args,
                    question_item=question_item,
                    condition=condition,
                    sample_number=sample_number,
                    pair_seed=pair_seed,
                    run_root=run_root,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
