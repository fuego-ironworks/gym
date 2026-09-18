#!/usr/bin/env python3
"""Maintain complete local Git mirrors from a provider-neutral manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class MirrorError(Exception):
    pass


@dataclass(frozen=True)
class MirrorSpec:
    relative_path: str
    remote: str


def run_git(
    *args: str,
    cwd: Path | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=capture,
        text=True,
    )


def validate_relative_path(text: str) -> str:
    path = PurePosixPath(text)
    if not text or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise MirrorError(f"invalid mirror path: {text!r}")
    if "\\" in text:
        raise MirrorError(f"mirror path must use '/' separators: {text!r}")
    return text


def load_manifest(path: Path) -> list[MirrorSpec]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MirrorError(str(error)) from error

    specs: list[MirrorSpec] = []
    seen: set[str] = set()

    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise MirrorError(
                f"{path}:{line_number}: expected RELATIVE_PATH<TAB>REMOTE"
            )
        relative_path = validate_relative_path(fields[0])
        remote = fields[1]
        if relative_path in seen:
            raise MirrorError(
                f"{path}:{line_number}: duplicate mirror path {relative_path!r}"
            )
        seen.add(relative_path)
        specs.append(MirrorSpec(relative_path, remote))

    if not specs:
        raise MirrorError(f"{path}: no mirror entries")
    return specs


def require_success(result: subprocess.CompletedProcess[str], action: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise MirrorError(f"{action} failed: {detail}")
    return result.stdout.strip()


def inspect_mirror(destination: Path, expected_remote: str) -> tuple[int, str]:
    bare = require_success(
        run_git("-C", str(destination), "rev-parse", "--is-bare-repository"),
        f"inspect {destination}",
    )
    if bare != "true":
        raise MirrorError(f"{destination}: not a bare Git repository")

    remote = require_success(
        run_git("-C", str(destination), "config", "--get", "remote.origin.url"),
        f"read origin for {destination}",
    )
    if remote != expected_remote:
        raise MirrorError(
            f"{destination}: origin does not match manifest; refusing to retarget"
        )

    refspec = require_success(
        run_git("-C", str(destination), "config", "--get-all", "remote.origin.fetch"),
        f"read mirror refspec for {destination}",
    ).splitlines()
    if refspec != ["+refs/*:refs/*"]:
        raise MirrorError(
            f"{destination}: origin fetch refspec is not a complete mirror"
        )

    require_success(
        run_git("-C", str(destination), "fsck", "--full"),
        f"git fsck for {destination}",
    )

    refs = require_success(
        run_git(
            "-C",
            str(destination),
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs",
        ),
        f"enumerate refs for {destination}",
    )
    lines = [line for line in refs.splitlines() if line]
    canonical = ("\n".join(lines) + ("\n" if lines else "")).encode()
    return len(lines), hashlib.sha256(canonical).hexdigest()


def update_one(root: Path, spec: MirrorSpec) -> tuple[int, str]:
    destination = root / Path(spec.relative_path)

    if destination.exists():
        inspect_mirror(destination, spec.remote)
        require_success(
            run_git("-C", str(destination), "remote", "update", "--prune"),
            f"update {destination}",
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = run_git(
            "clone",
            "--mirror",
            "--",
            spec.remote,
            str(destination),
        )
        require_success(result, f"clone {spec.relative_path}")

    return inspect_mirror(destination, spec.remote)


def verify_one(root: Path, spec: MirrorSpec) -> tuple[int, str]:
    destination = root / Path(spec.relative_path)
    if not destination.exists():
        raise MirrorError(f"{destination}: mirror is missing")
    return inspect_mirror(destination, spec.remote)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Maintain complete local Git mirrors from a manifest."
    )
    parser.add_argument("mode", choices=("update", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv[1:])

    try:
        specs = load_manifest(args.manifest)
        args.root.mkdir(parents=True, exist_ok=True)

        for spec in specs:
            if args.mode == "update":
                ref_count, refs_sha256 = update_one(args.root, spec)
            else:
                ref_count, refs_sha256 = verify_one(args.root, spec)
            print(
                f"PASS mode={args.mode} path={spec.relative_path} "
                f"refs={ref_count} refs_sha256={refs_sha256}"
            )
    except (MirrorError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
