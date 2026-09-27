#!/usr/bin/env python3
"""Run a separate offline Blackball-context protocol against local Ollama.

Conditions:
  without-blackball: literal question
  blackball-context: deterministic text read from an exact Blackball commit

This is intentionally separate from run_ollama.py.  It is not the URL-only
A/B protocol and never treats a URL string as repository access.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pathlib
import re
import subprocess
import sys
import time

from run_ollama import (
    call_ollama,
    load_questions,
    question_metadata,
    safe_component,
    utc_now,
    write_json,
)

BLACKBALL_REPOSITORY = "bl4ckb4ll/blackball"
CONDITIONS = ("without-blackball", "blackball-context")
PROTOCOL = "offline-blackball-context-v1"


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. qwen3:4b")
    parser.add_argument("--samples", type=int, default=5, help="paired samples per question")
    parser.add_argument("--questions", type=pathlib.Path, default=here / "questions.jsonl")
    parser.add_argument("--results", type=pathlib.Path, default=here / "context-results")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1000, help="base seed; sample N uses seed+N")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--blackball-repo",
        type=pathlib.Path,
        default=None,
        help="local Blackball git checkout; defaults to the checkout containing this script",
    )
    parser.add_argument(
        "--blackball-commit",
        required=True,
        help="exact 40-hex Blackball commit to freeze; symbolic refs are rejected",
    )
    parser.add_argument(
        "--context-path",
        action="append",
        required=True,
        help="UTF-8 repository file to inject; repeat for a deterministic subset",
    )
    return parser.parse_args()


def git_bytes(repo: pathlib.Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)}: {message}")
    return completed.stdout


def git_text(repo: pathlib.Path, *args: str) -> str:
    return git_bytes(repo, *args).decode("utf-8").strip()


def repo_root(candidate: pathlib.Path | None) -> pathlib.Path:
    start = candidate or pathlib.Path(__file__).resolve().parent
    return pathlib.Path(git_text(start, "rev-parse", "--show-toplevel"))


def freeze_commit(repo: pathlib.Path, requested: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", requested) is None:
        raise ValueError("--blackball-commit must be an exact 40-hex commit SHA")
    resolved = git_text(repo, "rev-parse", "--verify", f"{requested}^{{commit}}")
    if resolved.lower() != requested.lower():
        raise ValueError(f"commit resolved unexpectedly: {requested} -> {resolved}")
    return resolved.lower()


def normalize_context_paths(values: list[str]) -> list[str]:
    paths: set[str] = set()
    for raw in values:
        value = raw.strip()
        pure = pathlib.PurePosixPath(value)
        if not value or value == "." or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"invalid repository context path: {raw!r}")
        paths.add(pure.as_posix())
    return sorted(paths)


def build_context(repo: pathlib.Path, commit: str, paths: list[str]) -> dict:
    serialized = bytearray()
    serialized.extend(f"BLACKBALL REPOSITORY CONTEXT\ncommit: {commit}\n".encode("utf-8"))
    files: list[dict] = []

    for path in normalize_context_paths(paths):
        object_sha = git_text(repo, "rev-parse", f"{commit}:{path}")
        object_type = git_text(repo, "cat-file", "-t", object_sha)
        if object_type != "blob":
            raise ValueError(f"context path is not a file at {commit}: {path}")
        raw = git_bytes(repo, "cat-file", "-p", object_sha)
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"context path is not UTF-8 text: {path}") from exc

        serialized.extend(f"\n===== BEGIN BLACKBALL FILE: {path} =====\n".encode("utf-8"))
        serialized.extend(raw)
        if not raw.endswith(b"\n"):
            serialized.extend(b"\n")
        serialized.extend(f"===== END BLACKBALL FILE: {path} =====\n".encode("utf-8"))
        files.append({"path": path, "blob_sha": object_sha, "bytes": len(raw)})

    context_bytes = bytes(serialized)
    return {
        "text": context_bytes.decode("utf-8"),
        "sha256": hashlib.sha256(context_bytes).hexdigest(),
        "bytes": len(context_bytes),
        "files": files,
    }


def effective_prompt(question: str, condition: str, context_text: str) -> str:
    if condition == "without-blackball":
        return question
    if condition == "blackball-context":
        return f"{context_text}\n{question}"
    raise ValueError(condition)


def context_metadata(condition: str, commit: str, context: dict) -> dict:
    if condition != "blackball-context":
        return {
            "blackball_repository": None,
            "blackball_commit": None,
            "context_sha256": None,
            "context_bytes": None,
            "context_files": None,
        }
    return {
        "blackball_repository": BLACKBALL_REPOSITORY,
        "blackball_commit": commit,
        "context_sha256": context["sha256"],
        "context_bytes": context["bytes"],
        "context_files": context["files"],
    }


def run_one(
    *,
    args: argparse.Namespace,
    question_item: dict,
    condition: str,
    sample_number: int,
    pair_seed: int,
    run_root: pathlib.Path,
    commit: str,
    context: dict,
) -> None:
    question = question_item["question"]
    prompt = effective_prompt(question, condition, context["text"])
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
                    "protocol": PROTOCOL,
                    "condition": condition,
                    "effective_prompt": prompt,
                    **context_metadata(condition, commit, context),
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
                "protocol": PROTOCOL,
                "condition": condition,
                "effective_prompt": prompt,
                **context_metadata(condition, commit, context),
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

    try:
        repo = repo_root(args.blackball_repo)
        commit = freeze_commit(repo, args.blackball_commit)
        context = build_context(repo, commit, args.context_path)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

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
                "protocol": PROTOCOL,
                "model": args.model,
                "conditions": list(CONDITIONS),
                "samples": args.samples,
                "base_seed": args.seed,
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_predict": args.num_predict,
                },
                "blackball_repository": BLACKBALL_REPOSITORY,
                "blackball_commit": commit,
                "context_sha256": context["sha256"],
                "context_bytes": context["bytes"],
                "context_files": context["files"],
                "runner": "run_ollama_context.py",
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
                    commit=commit,
                    context=context,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
