#!/usr/bin/env python3
"""Run reproducible GPT-OSS source-comprehension pairs from real Git revisions."""
from __future__ import annotations

import argparse
import datetime as datetime
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cases" / "functorial_c" / "pairs.json"


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_manifest(path: Path = MANIFEST) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or not spec.get("pairs"):
        raise ValueError("Missing/nonempty schema_version=1 pairs")
    pair_ids = set()
    for pair in spec["pairs"]:
        if (not re.fullmatch(r"[a-z0-9-]+", pair["id"])
                or pair["id"] in pair_ids
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", pair["repository"])):
            raise ValueError("Invalid/duplicate pair ID or repository")
        pair_ids.add(pair["id"])
        for phase in ("before", "after"):
            version = pair[phase]
            if not re.fullmatch(r"[0-9a-f]{40}", version["commit"]):
                raise ValueError("Unpinned commit")
            aliases = set()
            for f in version["files"]:
                path = PurePosixPath(f["path"])
                if (path.is_absolute() or ".." in path.parts
                        or not re.fullmatch(r"[0-9a-f]{40}", f["blob_sha"])
                        or f["alias"] not in ("code.c", "seifert.h")
                        or f["alias"] in aliases):
                    raise ValueError("Unpinned or unsafe source path")
                aliases.add(f["alias"])
            if "code.c" not in aliases:
                raise ValueError("Source code is missing")
        ids = [x["id"] for x in pair.get("questions", []) + pair.get("edits", [])]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Missing or duplicate task IDs")
    return spec


def read_sources(pair: dict, phase: str, cache: Path, offline: bool) -> dict[str, str]:
    sources = {}
    for item in pair[phase]["files"]:
        dest = cache / pair["id"] / phase / item["alias"]
        if dest.exists():
            data = dest.read_bytes()
        else:
            if offline:
                raise FileNotFoundError(f"Missing source cache: {dest}")
            url = ("https://raw.githubusercontent.com/" + pair["repository"]
                   + "/" + pair[phase]["commit"] + "/"
                   + urllib.parse.quote(item["path"], safe="/"))
            with urllib.request.urlopen(url, timeout=45) as stream:
                data = stream.read(160001)
            if len(data) > 160000:
                raise ValueError("Source exceeds 160 KB cap")
        if blob_hash(data) != item["blob_sha"]:
            raise ValueError(f"Git blob mismatch in {pair['id']} {phase} {item['alias']}")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        sources[item["alias"]] = data.decode("utf-8")
    return sources


def make_prompt(sources: dict, task: dict, kind: str) -> str:
    message = [
        "Read the actual C files below. Use function bodies and control flow, not names alone.",
        "Some C code uses ← for assignment, × for multiplication and ÷ for division.",
        "The files' historical revision and style category are deliberately not supplied.",
    ]
    for alias, code in sources.items():
        message += ["FILE " + alias + ":", "~~~c", code, "~~~"]
    if kind == "question":
        message += ["QUESTION: " + task["question"],
                    'Reply with exactly one JSON object: {"answer": VALUE}.',
                    'Use {"answer": null} only if the result cannot be determined.',
                    "No explanation or additional fields."]
    else:
        message += ["EDIT REQUEST: " + task["instruction"],
                    "Reply with a unified diff for code.c only.",
                    "Use --- a/code.c and +++ b/code.c headers and @@ hunks.",
                    "No explanation or other files."]
    return "\n".join(message)


def equal_answer(actual, expected) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equal_answer(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equal_answer(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def check_answer(content: str, expected) -> dict:
    try:
        obj = json.loads(content.strip())
        if not isinstance(obj, dict) or set(obj) != {"answer"}:
            raise ValueError("Incorrect JSON structure")
        return {"parseable": True, "correct": equal_answer(obj["answer"], expected)}
    except (ValueError, json.JSONDecodeError):
        return {"parseable": False, "correct": False}


def check_patch(content: str, original: str) -> dict:
    patch = content.strip()
    fence = re.escape(chr(96) * 3)
    match = re.fullmatch(fence + r"(?:diff|patch)?\s*\n(.*?)\n" + fence,
                         patch, flags=re.S)
    if match:
        patch = match.group(1)
    patch = patch.strip() + "\n"
    if (re.findall(r"(?m)^--- ([^\n]+)$", patch) != ["a/code.c"]
            or re.findall(r"(?m)^\+\+\+ ([^\n]+)$", patch) != ["b/code.c"]
            or "@@" not in patch):
        return {"applies": False, "semantic_correctness": "NOT_CHECKED",
                "error": "Not a single-file code.c unified diff"}
    with tempfile.TemporaryDirectory() as temporary:
        (Path(temporary) / "code.c").write_text(original, encoding="utf-8")
        try:
            p = subprocess.run(["git", "apply", "--check", "-p1", "-"],
                               input=patch, text=True, capture_output=True,
                               cwd=temporary, timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"applies": False, "semantic_correctness": "NOT_CHECKED",
                    "error": repr(exc)}
    return {"applies": p.returncode == 0, "semantic_correctness": "NOT_CHECKED",
            "error": p.stderr.strip()[:500]}


def model_identity(endpoint: str, model: str) -> dict:
    if not endpoint.endswith("/api/chat"):
        raise ValueError("Only Ollama's /api/chat adapter is supported")
    with urllib.request.urlopen(endpoint[:-4] + "tags", timeout=12) as response:
        tags = json.load(response)
    matches = [m for m in tags.get("models", []) if model in (m.get("name"), m.get("model"))]
    if not matches:
        raise RuntimeError(f"{model} not installed; run: ollama pull {model}")
    m = matches[0]
    return {k: m.get(k) for k in ("name", "digest", "size", "modified_at", "details")}


def ask_model(endpoint: str, model: str, prompt: str, kind: str, seed: int) -> dict:
    data = {"model": model, "stream": False, "messages": [
        {"role": "system", "content": "Answer the precise code question or edit request."},
        {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": seed, "num_ctx": 16384}}
    if kind == "question":
        data["format"] = "json"
    request = urllib.request.Request(
        endpoint, data=json.dumps(data, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        out = json.load(response)
    out["_wall_seconds"] = round(time.monotonic() - start, 3)
    return out


def summarize(rows: list[dict]) -> None:
    totals = defaultdict(lambda: {"attempts": 0, "complete": 0, "passed": 0})
    for row in rows:
        if row.get("kind") not in ("question", "edit"):
            continue
        key = (row["pair"], row["kind"], row["phase"])
        cell = totals[key]
        cell["attempts"] += 1
        if row.get("status") == "completed":
            cell["complete"] += 1
            cell["passed"] += int(bool(row["score"].get(
                "correct" if row["kind"] == "question" else "applies")))
    for (pair, kind, phase), n in sorted(totals.items()):
        measure = "correct answers" if kind == "question" else "applicable patches ONLY"
        print(f"{pair} {phase} {kind}: {n['passed']}/{n['complete']} {measure}; "
              f"{n['attempts']-n['complete']} inference errors")


def main(argv: list[str] | None = None) -> int:
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--model", default="gpt-oss:20b")
    arg.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    arg.add_argument("--trials", type=int, default=3)
    arg.add_argument("--seed", type=int, default=20261009)
    arg.add_argument("--pair", action="append")
    arg.add_argument("--kind", choices=("both", "question", "edit"), default="both")
    arg.add_argument("--cache", type=Path, default=ROOT / ".cache" / "functorial-c")
    arg.add_argument("--offline", action="store_true")
    arg.add_argument("--dry-run", action="store_true", help="Verify source SHA only; no model calls")
    arg.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "functorial-c")
    arg.add_argument("--summarize", type=Path, help="Recompute summary from receipt JSONL")
    a = arg.parse_args(argv)
    if a.summarize:
        summarize([json.loads(line) for line in a.summarize.read_text().splitlines()])
        return 0
    if not 1 <= a.trials <= 100:
        arg.error("trials must be between 1 and 100")
    spec = read_manifest()
    pairs = [p for p in spec["pairs"] if not a.pair or p["id"] in a.pair]
    if not pairs or (a.pair and set(a.pair) != {p["id"] for p in pairs}):
        arg.error("Unknown pair ID")
    sources = {(pair["id"], phase): read_sources(pair, phase, a.cache, a.offline)
               for pair in pairs for phase in ("before", "after")}
    for pair, phase in sources:
        print(f"Verified actual historical sources: {pair}, {phase}")
    if a.dry_run:
        print("Model NOT_RUN: source-only preflight completed")
        return 0
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = a.output_dir / (run_id + ".jsonl")
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    try:
        identity = model_identity(a.endpoint, a.model)
    except Exception as exc:
        receipt.write_text(json.dumps({"run_id": run_id, "kind": "preflight",
                         "status": "BLOCKED", "reason": repr(exc),
                         "model": a.model, "manifest_sha256": manifest_hash}) + "\n")
        print(f"GPT-OSS unavailable, NOT_RUN. Receipt: {receipt}: {exc}", file=sys.stderr)
        return 2
    jobs = [
        (pair, task, kind, trial, phase)
        for pair in pairs
        for kind, label in (("question", "questions"), ("edit", "edits"))
        if a.kind in ("both", kind)
        for task in pair.get(label, [])
        for trial in range(a.trials)
        for phase in ("before", "after")
    ]
    random.Random(a.seed).shuffle(jobs)
    rows = []
    with receipt.open("x", encoding="utf-8") as stream:
        for pair, task, kind, trial, phase in jobs:
            source = sources[(pair["id"], phase)]
            prompt = make_prompt(source, task, kind)
            row = {
                "run_id": run_id, "time_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "pair": pair["id"], "phase": phase, "task_id": task["id"],
                "kind": kind, "trial": trial, "model": identity,
                "settings": {"temperature": 0, "seed": a.seed + trial, "num_ctx": 16384},
                "repository": pair["repository"], "commit": pair[phase]["commit"],
                "blobs": {f["alias"]: f["blob_sha"] for f in pair[phase]["files"]},
                "manifest_sha256": manifest_hash, "prompt": prompt}
            try:
                out = ask_model(a.endpoint, a.model, prompt, kind, a.seed + trial)
                content = out.get("message", {}).get("content", "")
                row.update(status="completed", response=content,
                           thinking=out.get("message", {}).get("thinking"),
                           timing={k: out.get(k) for k in (
                               "_wall_seconds", "eval_count", "eval_duration",
                               "prompt_eval_count", "total_duration") if k in out},
                           score=(check_answer(content, task["answer"]) if kind == "question"
                                  else check_patch(content, source["code.c"])))
            except Exception as exc:
                row.update(status="INFERENCE_ERROR", error=repr(exc))
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            rows.append(row)
            print(f"{row['pair']} {row['task_id']} {phase} trial {trial}: "
                  f"{row['status']} {row.get('score', {})}")
    summarize(rows)
    print("Saved independent full model receipts:", receipt)
    return 0 if all(r["status"] == "completed" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
