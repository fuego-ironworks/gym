from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import uuid
from importlib import metadata
import os
import platform
import sys
from pathlib import Path

# This existing Python entrypoint remains migration debt. New shared orchestration
# belongs in AICI; this change only hardens the current Gym consumer.
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_ROOT = Path(__file__).resolve().parents[2]


def require_commit(value: str, name: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable 40-hex commit")
    return value


def source_receipt(root: Path = REPO_ROOT) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    actual = require_commit(git("rev-parse", "HEAD"), "checked-out source")
    expected = os.environ.get("GYM_SOURCE_SHA")
    if expected is not None and require_commit(expected, "expected source") != actual:
        raise ValueError("checked-out source differs from the expected source head")
    # Untracked outputs are allowed, but all tracked source changes fail closed.
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked source has uncommitted changes")
    return {"commit_sha": actual, "expected_sha": expected, "tracked_tree_clean": True}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def atomic_json(path: Path, value: dict) -> None:
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_args(args: argparse.Namespace) -> None:
    if args.rank != 1:
        raise ValueError("the mechanical acceptance case requires rank 1")
    if args.steps < 0 or args.threads < 1:
        raise ValueError("steps must be non-negative and threads must be positive")
    if not 0 <= args.seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    if not args.example.strip() or not args.target.strip():
        raise ValueError("example and target must be nonempty")
    for name in ("alpha", "learning_rate"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def build_run_plan(args: argparse.Namespace, source: dict, model_source: dict,
                   environment: dict) -> dict:
    validate_args(args)
    if source.get("tracked_tree_clean") is not True:
        raise ValueError("source is not a verified clean checkout")
    if model_source["repository"] != args.model:
        raise ValueError("resolved model repository differs from the requested model")
    coordinates = {
        "source_commit": require_commit(source["commit_sha"], "source"),
        "model": args.model,
        "model_commit": require_commit(model_source["commit_sha"], "model"),
        "target": args.target, "rank": args.rank, "alpha": args.alpha,
        "optimizer": "SGD", "learning_rate": args.learning_rate,
        "optimizer_steps": args.steps, "seed": args.seed,
        "example": args.example, "device": "cpu", "dtype": "float32",
        "threads": args.threads, "environment": environment,
    }
    identity = {"schema": 1, "kind": "gym-lora-mechanics", "coordinates": coordinates}
    return {**identity, "job_id": hashlib.sha256(canonical_json(identity)).hexdigest(),
            "paid_execution_allowed": False}


def require_same_plan(expected: dict, actual: dict) -> None:
    # A plan is not a smoke-test receipt and never authorizes a paid request.
    if canonical_json(expected) != canonical_json(actual):
        raise ValueError("plan is missing, stale, or belongs to a different job")


def reserve_run_directory(root: Path, job_id: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", job_id) is None:
        raise ValueError("invalid job identity")
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if run_id is not None or attempt is not None:
        if not (run_id and attempt and run_id.isascii() and run_id.isdigit()
                and attempt.isascii() and attempt.isdigit()):
            raise ValueError("invalid or incomplete GitHub run identity")
        invocation = f"github-{run_id}-attempt-{attempt}"
    else:
        invocation = f"local-{uuid.uuid4().hex}"
    directory = root / f"{job_id}-{invocation}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Exclusive: never overwrite a previous attempt.
    return directory


def load_pinned_model(model_source: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    commit = require_commit(model_source["commit_sha"], "model")
    model_name = model_source["repository"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=commit, trust_remote_code=False, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=commit, trust_remote_code=False
    )
    return model, tokenizer


def hf_source_receipt(model_name: str, revision: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_name, revision=revision, files_metadata=True)
    commit = require_commit(info.sha, "resolved model")
    if COMMIT_RE.fullmatch(revision) is not None and commit != revision:
        raise ValueError("model host returned a different commit than requested")
    files = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        lfs = getattr(sibling, "lfs", None)
        lfs_sha256 = None
        if isinstance(lfs, dict):
            lfs_sha256 = lfs.get("sha256")
        elif lfs is not None:
            lfs_sha256 = getattr(lfs, "sha256", None)
        files.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": lfs_sha256,
            }
        )
    return {
        "repository": model_name,
        "requested_revision": revision,
        "commit_sha": commit,
        "files": files,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_index(output_dir: Path) -> None:
    entries = []
    for path in sorted(output_dir.iterdir()):
        if path.name in {"artifact-index.json", "COMPLETE.json", "FAILED.json"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected artifact type: {path.name}")
        entries.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    atomic_json(output_dir / "artifact-index.json", {"schema": 1, "files": entries})


def verify_artifact_index(output_dir: Path) -> None:
    index = json.loads((output_dir / "artifact-index.json").read_text())
    if index.get("schema") != 1 or not index.get("files"):
        raise ValueError("artifact index is empty or has an unsupported schema")
    names = set()
    for row in index["files"]:
        name = row["path"]
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in {".", "..", "artifact-index.json", "COMPLETE.json", "FAILED.json"}
                or name in names):
            raise ValueError("invalid or duplicate artifact path")
        names.add(name)
        path = output_dir / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"]
                or file_sha256(path) != row["sha256"]):
            raise ValueError(f"missing or corrupt artifact: {name}")
    actual = {p.name for p in output_dir.iterdir()
              if p.name not in {"artifact-index.json", "COMPLETE.json", "FAILED.json"}}
    if names != actual:
        raise ValueError("artifact index does not cover the complete output set")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stepwise rank-one LoRA differential tests against a Pythia checkpoint."
    )
    parser.add_argument("--model", default="EleutherAI/pythia-14m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument(
        "--target",
        default="gpt_neox.layers.0.attention.query_key_value",
        help="Fully qualified nn.Linear module to wrap with the adapter.",
    )
    parser.add_argument("--example", default="The moon orbits the Earth.")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true",
                        help="Resolve sources and write a plan; do not load weights or train")
    parser.add_argument("--expected-plan", type=Path,
                        help="Require exact equality to a previously reviewed plan.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm/models/runs/pythia-14m-lora-inspectability"),
    )
    return parser.parse_args(argv)


def execute_plan(args: argparse.Namespace, directory: Path, source: dict,
                 model_source: dict, plan: dict) -> dict:
    import torch
    import huggingface_hub
    from lora_differential import attach_lora, run_steps, run_test0, write_run_manifest

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, tokenizer = load_pinned_model(model_source)
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    input_ids = tokenizer(args.example, return_tensors="pt",
                          add_special_tokens=False)["input_ids"]
    if input_ids.shape[-1] < 2:
        raise ValueError("training example must tokenize to at least two tokens")
    labels = input_ids.clone()
    atomic_json(directory / "inputs.json", {
        "input_ids": input_ids.tolist(), "labels": labels.tolist(),
        "loss_policy": "all next-token positions, unchanged from the existing harness",
    })
    adapter = attach_lora(model, args.target, rank=args.rank,
                          alpha=args.alpha, seed=args.seed)
    test0 = run_test0(model, adapter, input_ids, labels, directory)
    if test0["passed"] is not True:
        raise ValueError("zero-update acceptance failed; training was not started")
    steps = run_steps(model, adapter, input_ids, labels, directory,
                      steps=args.steps, learning_rate=args.learning_rate)
    if len(steps) != args.steps or any(item["passed"] is not True for item in steps):
        raise ValueError("optimizer-step acceptance failed or receipts are incomplete")
    manifest = write_run_manifest(
        directory, model_name=args.model, revision=model_source["commit_sha"],
        target=args.target, rank=args.rank, alpha=args.alpha,
        learning_rate=args.learning_rate, steps=args.steps, seed=args.seed,
        example=args.example, test0=test0, step_receipts=steps,
    )
    manifest["job_id"] = plan["job_id"]
    manifest["model_source"] = model_source
    manifest["source"] = source
    manifest["paid_execution_allowed"] = False
    manifest["runtime"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": metadata.version("transformers"),
        "huggingface_hub": huggingface_hub.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "source_sha": source["commit_sha"],
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "github_workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    source = source_receipt()
    # Resolve BEFORE loading. All subsequent readers use this same immutable SHA.
    model_source = hf_source_receipt(args.model, args.revision)
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: metadata.version(name)
                     for name in ("torch", "transformers", "huggingface_hub")},
    }
    plan = build_run_plan(args, source, model_source, environment)
    if args.expected_plan is not None:
        expected = json.loads(args.expected_plan.read_text(encoding="utf-8"))
        require_same_plan(expected, plan)
    directory = reserve_run_directory(args.output_dir, plan["job_id"])
    atomic_json(directory / "plan.json", plan)
    atomic_json(directory / "sources.json", {"source": source, "model_source": model_source})
    if args.plan_only:
        print(json.dumps({"output_dir": str(directory), "status": "PLANNED",
                          "job_id": plan["job_id"], "paid_execution_allowed": False}))
        return 0
    try:
        manifest = execute_plan(args, directory, source, model_source, plan)
        if manifest.get("passed") is not True:
            raise ValueError("run did not produce passing acceptance")
        write_artifact_index(directory)
        verify_artifact_index(directory)
        # A partial write or a successful --plan-only cannot manufacture completion.
        atomic_json(directory / "COMPLETE.json", {
            "job_id": plan["job_id"], "passed": True,
            "artifact_index_sha256": file_sha256(directory / "artifact-index.json"),
            "paid_execution_allowed": False,
        })
    except BaseException as error:
        atomic_json(directory / "FAILED.json", {
            "job_id": plan["job_id"], "passed": False,
            "error_type": type(error).__name__, "paid_execution_allowed": False,
        })
        raise
    print(json.dumps({"output_dir": str(directory), "passed": True,
                      "job_id": plan["job_id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
