from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_github_metadata  # noqa: E402


def fake_gh_run(
    command: list[str],
    check: bool,
    capture_output: bool,
    text: bool,
) -> subprocess.CompletedProcess[str]:
    del check, capture_output, text
    endpoint = command[-1]
    paginated = "--paginate" in command

    if endpoint.startswith("/repos/") and endpoint.count("/") == 3:
        value: object = {"kind": "repository", "url": endpoint}
    elif endpoint.endswith("/issues/7"):
        value = {"kind": "issue", "number": 7}
    elif endpoint.endswith("/pulls/9"):
        value = {"kind": "pull", "number": 9}
    elif endpoint.endswith("/releases/11"):
        value = {"kind": "release", "id": 11}
    elif endpoint.endswith("/actions/runs/13"):
        value = {"kind": "run", "id": 13}
    elif paginated:
        value = [[{"endpoint": endpoint, "page": 1}]]
    else:
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr=f"unexpected endpoint: {endpoint}",
        )

    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps(value),
        stderr="",
    )


class GithubExportTests(unittest.TestCase):
    def test_exports_selected_metadata_and_hashes_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "export.tsv"
            output = root / "snapshot"
            spec.write_text(
                "issue\texample/project\t7\n"
                "pull\texample/project\t9\n"
                "release\texample/project\t11\n"
                "workflow_run\texample/project\t13\n",
                encoding="utf-8",
            )

            with patch.object(
                export_github_metadata.subprocess,
                "run",
                side_effect=fake_gh_run,
            ):
                status = export_github_metadata.main(
                    ["export_github_metadata.py", str(spec), str(output)]
                )

            self.assertEqual(0, status)
            snapshot = json.loads(
                (output / "snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual("gym.github_export.v1", snapshot["schema_version"])
            self.assertEqual(4, len(snapshot["exports"]))

            required = {
                "github/example/project/repository.json",
                "github/example/project/issues/7/issue.json",
                "github/example/project/issues/7/comments.json",
                "github/example/project/pulls/9/pull.json",
                "github/example/project/pulls/9/files.json",
                "github/example/project/releases/11/release.json",
                "github/example/project/workflow-runs/13/run.json",
                "github/example/project/workflow-runs/13/jobs.json",
                "github/example/project/workflow-runs/13/artifacts.json",
            }
            recorded = {entry["path"] for entry in snapshot["files"]}
            self.assertTrue(required.issubset(recorded))

            for entry in snapshot["files"]:
                payload = (output / entry["path"]).read_bytes()
                self.assertEqual(len(payload), entry["bytes"])
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    entry["sha256"],
                )

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "export.tsv"
            output = root / "snapshot"
            spec.write_text(
                "issue\texample/project\t7\n",
                encoding="utf-8",
            )
            output.mkdir()

            with patch.object(
                export_github_metadata.subprocess,
                "run",
                side_effect=fake_gh_run,
            ):
                status = export_github_metadata.main(
                    ["export_github_metadata.py", str(spec), str(output)]
                )

            self.assertEqual(1, status)
            self.assertEqual([], list(output.iterdir()))

    def test_spec_rejects_unsafe_or_duplicate_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "export.tsv"

            spec.write_text(
                "issue\t../project\t7\n",
                encoding="utf-8",
            )
            unsafe = export_github_metadata.main(
                ["export_github_metadata.py", str(spec), str(root / "unsafe")]
            )
            self.assertEqual(1, unsafe)

            spec.write_text(
                "issue\texample/project\t7\n"
                "issue\texample/project\t7\n",
                encoding="utf-8",
            )
            duplicate = export_github_metadata.main(
                ["export_github_metadata.py", str(spec), str(root / "duplicate")]
            )
            self.assertEqual(1, duplicate)

    def test_api_failure_removes_partial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "export.tsv"
            output = root / "snapshot"
            spec.write_text(
                "issue\texample/project\t7\n",
                encoding="utf-8",
            )

            def fail_comments(
                command: list[str],
                check: bool,
                capture_output: bool,
                text: bool,
            ) -> subprocess.CompletedProcess[str]:
                if command[-1].endswith("/issues/7/comments?per_page=100"):
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr="simulated API failure",
                    )
                return fake_gh_run(command, check, capture_output, text)

            with patch.object(
                export_github_metadata.subprocess,
                "run",
                side_effect=fail_comments,
            ):
                status = export_github_metadata.main(
                    ["export_github_metadata.py", str(spec), str(output)]
                )

            self.assertEqual(1, status)
            self.assertFalse(output.exists())
            self.assertEqual(
                [],
                list(root.glob("snapshot.partial.*")),
            )


if __name__ == "__main__":
    unittest.main()
