#!/usr/bin/env python3
"""Upload completed dense NeoX checkpoints and retire local copies safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

READY_MARKER = ".gym-ready"
STOP_STEP = 1000


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileDigest:
    path: str
    size: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    prefix = "global_step"
    if not path.name.startswith(prefix):
        raise CheckpointError(f"not a NeoX checkpoint directory: {path}")
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit():
        raise CheckpointError(f"bad checkpoint step: {path.name}")
    return int(suffix)


def checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not (checkpoint_dir / READY_MARKER).is_file():
        raise CheckpointError(f"checkpoint is not ready: {checkpoint_dir}")

    files: list[Path] = []
    for path in checkpoint_dir.rglob("*"):
        if path.is_symlink():
            raise CheckpointError(f"refusing symlink in checkpoint: {path}")
        if not path.is_file() or path.name == READY_MARKER:
            continue
        if path.name.endswith("optim_states.pt"):
            raise CheckpointError(f"optimizer state survived slim save: {path}")
        files.append(path)
    if not files:
        raise CheckpointError(f"checkpoint has no payload files: {checkpoint_dir}")
    return sorted(files)


def file_manifest(checkpoint_dir: Path) -> list[FileDigest]:
    rows: list[FileDigest] = []
    for path in checkpoint_files(checkpoint_dir):
        rows.append(
            FileDigest(
                path=path.relative_to(checkpoint_dir).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return rows


def make_archive(checkpoint_dir: Path, staging_dir: Path) -> tuple[Path, list[FileDigest], str, int]:
    step = checkpoint_step(checkpoint_dir)
    manifest = file_manifest(checkpoint_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_path = staging_dir / f"global_step{step}.tar"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".global_step{step}.", suffix=".tar", dir=str(staging_dir)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tarfile.open(tmp_path, mode="w") as archive:
            for row in manifest:
                source = checkpoint_dir / row.path
                archive.add(
                    source,
                    arcname=f"global_step{step}/{row.path}",
                    recursive=False,
                )
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    size = final_path.stat().st_size
    return final_path, manifest, sha256_file(final_path), size


def _lfs_sha256(file_info: Any) -> Optional[str]:
    lfs = getattr(file_info, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return str(value) if value else None


class HuggingFaceHub:
    def __init__(self, repo_id: str, revision: str, token: Optional[str]) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise CheckpointError(
                "huggingface_hub is required for checkpoint upload"
            ) from exc
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.api = HfApi(token=token)

    def upload(self, local_path: Path, remote_path: str, message: str) -> str:
        result = self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="model",
            revision=self.revision,
            commit_message=message,
            token=self.token,
        )
        return str(result)

    def remote_lfs_sha256(self, remote_path: str) -> Optional[str]:
        info = self.api.model_info(
            self.repo_id,
            revision=self.revision,
            files_metadata=True,
            token=self.token,
        )
        for sibling in info.siblings:
            name = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
            if name == remote_path:
                return _lfs_sha256(sibling)
        return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def retire_checkpoint(
    checkpoint_dir: Path,
    *,
    receipts_dir: Path,
    staging_dir: Path,
    hub: Any,
    remote_prefix: str,
    retire: bool = True,
) -> dict[str, Any]:
    step = checkpoint_step(checkpoint_dir)
    if step > STOP_STEP:
        raise CheckpointError(f"refusing checkpoint after step {STOP_STEP}: {step}")

    archive, files, archive_sha256, archive_size = make_archive(
        checkpoint_dir, staging_dir
    )
    remote_archive = f"{remote_prefix.rstrip('/')}/checkpoints/{archive.name}"
    commit = hub.upload(
        archive,
        remote_archive,
        f"Upload dense Pythia reproduction checkpoint step {step}",
    )
    remote_sha256 = hub.remote_lfs_sha256(remote_archive)
    if remote_sha256 != archive_sha256:
        raise CheckpointError(
            f"remote hash mismatch for step {step}: "
            f"local={archive_sha256} remote={remote_sha256}"
        )

    receipt = {
        "schema": 1,
        "step": step,
        "checkpoint": checkpoint_dir.name,
        "archive": archive.name,
        "archive_size": archive_size,
        "archive_sha256": archive_sha256,
        "remote_path": remote_archive,
        "remote_lfs_sha256": remote_sha256,
        "upload_result": commit,
        "files": [row.__dict__ for row in files],
        "retired_local": bool(retire),
    }
    receipt_path = receipts_dir / f"step-{step:06d}.json"
    atomic_json(receipt_path, receipt)
    remote_receipt = f"{remote_prefix.rstrip('/')}/receipts/{receipt_path.name}"
    hub.upload(
        receipt_path,
        remote_receipt,
        f"Record dense Pythia reproduction checkpoint step {step}",
    )

    # Local bytes become disposable only after the archive hash matches the Hub
    # and the receipt has also been accepted by the Hub.
    if retire:
        shutil.rmtree(checkpoint_dir)
        archive.unlink()
    return receipt


def ready_checkpoints(save_dir: Path) -> Iterable[Path]:
    candidates = []
    for path in save_dir.glob("global_step*"):
        if not path.is_dir() or not (path / READY_MARKER).is_file():
            continue
        candidates.append((checkpoint_step(path), path))
    for _, path in sorted(candidates):
        yield path


def watch(
    *,
    save_dir: Path,
    receipts_dir: Path,
    staging_dir: Path,
    hub: Any,
    remote_prefix: str,
    poll_seconds: float,
    retire: bool,
    stop_step: int,
) -> None:
    completed: set[int] = set()
    while True:
        progress = False
        for checkpoint_dir in ready_checkpoints(save_dir):
            step = checkpoint_step(checkpoint_dir)
            if step in completed:
                continue
            receipt = retire_checkpoint(
                checkpoint_dir,
                receipts_dir=receipts_dir,
                staging_dir=staging_dir,
                hub=hub,
                remote_prefix=remote_prefix,
                retire=retire,
            )
            completed.add(step)
            progress = True
            print(
                f"uploaded step {step}: {receipt['archive_sha256']} "
                f"retired_local={receipt['retired_local']}",
                flush=True,
            )
            if step == stop_step:
                return
        if not progress:
            time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True, help="existing Hugging Face model repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--remote-prefix", default="dense-0-512")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stop-step", type=int, default=STOP_STEP)
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stop_step > STOP_STEP:
        raise CheckpointError(f"stop step may not exceed {STOP_STEP}")
    hub = HuggingFaceHub(args.repo, args.revision, os.environ.get("HF_TOKEN"))
    if args.once:
        found = list(ready_checkpoints(args.save_dir))
        if not found:
            print("no ready checkpoints")
            return 0
        for checkpoint_dir in found:
            retire_checkpoint(
                checkpoint_dir,
                receipts_dir=args.receipts_dir,
                staging_dir=args.staging_dir,
                hub=hub,
                remote_prefix=args.remote_prefix,
                retire=not args.keep_local,
            )
        return 0

    watch(
        save_dir=args.save_dir,
        receipts_dir=args.receipts_dir,
        staging_dir=args.staging_dir,
        hub=hub,
        remote_prefix=args.remote_prefix,
        poll_seconds=args.poll_seconds,
        retire=not args.keep_local,
        stop_step=args.stop_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
