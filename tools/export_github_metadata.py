#!/usr/bin/env python3
"""Export explicitly selected GitHub metadata into a local immutable snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
KINDS = {"issue", "pull", "release", "workflow_run"}


class ExportError(Exception):
    pass


@dataclass(frozen=True)
class ExportSpec:
    kind: str
    repository: str
    number: int


def load_spec(path: Path) -> list[ExportSpec]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExportError(str(error)) from error

    result: list[ExportSpec] = []
    seen: set[tuple[str, str, int]] = set()

    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ExportError(
                f"{path}:{line_number}: expected KIND<TAB>OWNER/REPO<TAB>NUMBER"
            )
        kind, repository, number_text = fields
        if kind not in KINDS:
            raise ExportError(f"{path}:{line_number}: unsupported kind {kind!r}")
        if REPO_RE.fullmatch(repository) is None:
            raise ExportError(
                f"{path}:{line_number}: invalid repository {repository!r}"
            )
        try:
            number = int(number_text)
        except ValueError as error:
            raise ExportError(
                f"{path}:{line_number}: invalid number {number_text!r}"
            ) from error
        if number <= 0:
            raise ExportError(f"{path}:{line_number}: number must be positive")

        key = (kind, repository, number)
        if key in seen:
            raise ExportError(f"{path}:{line_number}: duplicate export request")
        seen.add(key)
        result.append(ExportSpec(kind, repository, number))

    if not result:
        raise ExportError(f"{path}: no export requests")
    return result


def gh_json(endpoint: str, paginate: bool = False) -> object:
    command = ["gh", "api"]
    if paginate:
        command += ["--paginate", "--slurp"]
    command.append(endpoint)

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ExportError(f"gh api {endpoint} failed: {detail}")

    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ExportError(f"gh api {endpoint} returned invalid JSON") from error

    if paginate:
        if not isinstance(value, list):
            raise ExportError(f"gh api {endpoint} pagination was not a list")
        flattened: list[object] = []
        for page in value:
            if isinstance(page, list):
                flattened.extend(page)
            else:
                flattened.append(page)
        return flattened

    return value


def safe_repository_path(repository: str) -> Path:
    owner, name = repository.split("/", 1)
    return Path("github") / owner / name


def write_json(root: Path, relative: Path, value: object) -> dict[str, object]:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "path": relative.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def export_endpoints(
    root: Path,
    base: Path,
    endpoints: list[tuple[str, str, bool]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for filename, endpoint, paginate in endpoints:
        records.append(
            {
                **write_json(
                    root,
                    base / filename,
                    gh_json(endpoint, paginate=paginate),
                ),
                "endpoint": endpoint,
            }
        )
    return records


def export_repository(root: Path, repository: str) -> list[dict[str, object]]:
    base = safe_repository_path(repository)
    endpoint = f"/repos/{repository}"
    return [
        {
            **write_json(root, base / "repository.json", gh_json(endpoint)),
            "endpoint": endpoint,
        }
    ]


def export_issue(root: Path, spec: ExportSpec) -> list[dict[str, object]]:
    base = safe_repository_path(spec.repository) / "issues" / str(spec.number)
    return export_endpoints(
        root,
        base,
        [
            ("issue.json", f"/repos/{spec.repository}/issues/{spec.number}", False),
            (
                "comments.json",
                f"/repos/{spec.repository}/issues/{spec.number}/comments?per_page=100",
                True,
            ),
        ],
    )


def export_pull(root: Path, spec: ExportSpec) -> list[dict[str, object]]:
    base = safe_repository_path(spec.repository) / "pulls" / str(spec.number)
    return export_endpoints(
        root,
        base,
        [
            ("pull.json", f"/repos/{spec.repository}/pulls/{spec.number}", False),
            (
                "issue-comments.json",
                f"/repos/{spec.repository}/issues/{spec.number}/comments?per_page=100",
                True,
            ),
            (
                "reviews.json",
                f"/repos/{spec.repository}/pulls/{spec.number}/reviews?per_page=100",
                True,
            ),
            (
                "review-comments.json",
                f"/repos/{spec.repository}/pulls/{spec.number}/comments?per_page=100",
                True,
            ),
            (
                "commits.json",
                f"/repos/{spec.repository}/pulls/{spec.number}/commits?per_page=100",
                True,
            ),
            (
                "files.json",
                f"/repos/{spec.repository}/pulls/{spec.number}/files?per_page=100",
                True,
            ),
        ],
    )


def export_release(root: Path, spec: ExportSpec) -> list[dict[str, object]]:
    base = safe_repository_path(spec.repository) / "releases" / str(spec.number)
    endpoint = f"/repos/{spec.repository}/releases/{spec.number}"
    return [
        {
            **write_json(root, base / "release.json", gh_json(endpoint)),
            "endpoint": endpoint,
        }
    ]


def export_workflow_run(root: Path, spec: ExportSpec) -> list[dict[str, object]]:
    base = (
        safe_repository_path(spec.repository)
        / "workflow-runs"
        / str(spec.number)
    )
    return export_endpoints(
        root,
        base,
        [
            (
                "run.json",
                f"/repos/{spec.repository}/actions/runs/{spec.number}",
                False,
            ),
            (
                "jobs.json",
                f"/repos/{spec.repository}/actions/runs/{spec.number}/jobs?per_page=100",
                True,
            ),
            (
                "artifacts.json",
                f"/repos/{spec.repository}/actions/runs/{spec.number}/artifacts?per_page=100",
                True,
            ),
        ],
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Export selected GitHub metadata into a local snapshot."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv[1:])

    temporary = args.output.with_name(
        args.output.name + f".partial.{os.getpid()}"
    )

    try:
        if args.output.exists():
            raise ExportError(f"{args.output}: output already exists")
        if temporary.exists():
            raise ExportError(f"{temporary}: temporary output already exists")

        specs = load_spec(args.spec)
        repositories = sorted({spec.repository for spec in specs})
        temporary.mkdir(parents=True)

        files: list[dict[str, object]] = []
        for repository in repositories:
            files.extend(export_repository(temporary, repository))

        exports: list[dict[str, object]] = []
        for spec in specs:
            if spec.kind == "issue":
                new_files = export_issue(temporary, spec)
            elif spec.kind == "pull":
                new_files = export_pull(temporary, spec)
            elif spec.kind == "release":
                new_files = export_release(temporary, spec)
            else:
                new_files = export_workflow_run(temporary, spec)

            files.extend(new_files)
            exports.append(
                {
                    "kind": spec.kind,
                    "repository": spec.repository,
                    "number": spec.number,
                }
            )

        snapshot = {
            "schema_version": "gym.github_export.v1",
            "exports": exports,
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        write_json(temporary, Path("snapshot.json"), snapshot)
        temporary.rename(args.output)
        print(
            f"PASS exports={len(exports)} files={len(files)} output={args.output}"
        )
    except (ExportError, OSError) as error:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
