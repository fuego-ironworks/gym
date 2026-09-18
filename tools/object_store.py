#!/usr/bin/env python3
"""Store and verify immutable files by SHA-256 outside the Git repository."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_SIZE = 1024 * 1024


class ObjectStoreError(Exception):
    """Base class for object-store failures."""


class CorruptObjectError(ObjectStoreError):
    """An existing object does not match the digest encoded by its path."""


def require_sha256(text: str) -> str:
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError("SHA256 must be exactly 64 lowercase hexadecimal characters")
    return text


def object_path(store: Path, sha256: str) -> Path:
    digest = require_sha256(sha256)
    return store / "sha256" / digest[:2] / digest[2:]


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_object(
    store: Path,
    sha256: str,
    expected_size: int | None = None,
) -> tuple[bool, str]:
    path = object_path(store, sha256)
    try:
        actual_sha256, actual_size = hash_file(path)
    except FileNotFoundError:
        return False, f"MISSING sha256={sha256} path={path}"
    except OSError as error:
        return False, f"ERROR sha256={sha256} path={path}: {error}"

    if actual_sha256 != sha256:
        return (
            False,
            f"CORRUPT sha256={sha256} actual_sha256={actual_sha256} path={path}",
        )
    if expected_size is not None and actual_size != expected_size:
        return (
            False,
            f"SIZE_MISMATCH sha256={sha256} expected={expected_size} "
            f"actual={actual_size} path={path}",
        )
    return True, f"PASS sha256={sha256} size_bytes={actual_size} path={path}"


def put_file(store: Path, source: Path) -> tuple[str, int, Path, bool]:
    if not source.is_file():
        raise ObjectStoreError(f"source is not a regular file: {source}")

    store.mkdir(parents=True, exist_ok=True)
    temporary_directory = store / ".tmp"
    temporary_directory.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    digest = hashlib.sha256()
    size = 0

    try:
        with source.open("rb") as input_file:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="object-",
                dir=temporary_directory,
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                while True:
                    chunk = input_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())

        sha256 = digest.hexdigest()
        destination = object_path(store, sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            ok, message = verify_object(store, sha256, size)
            if not ok:
                raise CorruptObjectError(message)
            temporary_path.unlink()
            temporary_path = None
            return sha256, size, destination, False

        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            ok, message = verify_object(store, sha256, size)
            if not ok:
                raise CorruptObjectError(message)
            temporary_path.unlink()
            temporary_path = None
            return sha256, size, destination, False

        temporary_path.unlink()
        temporary_path = None
        return sha256, size, destination, True
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Store and verify immutable files by SHA-256."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    put_parser = subparsers.add_parser("put", help="copy one file into the object store")
    put_parser.add_argument("store", type=Path)
    put_parser.add_argument("file", type=Path)

    verify_parser = subparsers.add_parser("verify", help="verify one retained object")
    verify_parser.add_argument("store", type=Path)
    verify_parser.add_argument("sha256")
    verify_parser.add_argument("--size-bytes", type=int)

    path_parser = subparsers.add_parser("path", help="print the path for one digest")
    path_parser.add_argument("store", type=Path)
    path_parser.add_argument("sha256")

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    try:
        if args.command == "put":
            sha256, size, path, created = put_file(args.store, args.file)
            state = "STORED" if created else "PRESENT"
            print(f"{state} sha256={sha256} size_bytes={size} path={path}")
            return 0

        if args.command == "verify":
            if args.size_bytes is not None and args.size_bytes < 0:
                parser.error("--size-bytes must be >= 0")
            sha256 = require_sha256(args.sha256)
            ok, message = verify_object(args.store, sha256, args.size_bytes)
            print(message, file=sys.stdout if ok else sys.stderr)
            return 0 if ok else 1

        if args.command == "path":
            print(object_path(args.store, args.sha256))
            return 0
    except (OSError, ObjectStoreError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
