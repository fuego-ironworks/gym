from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRROR_TOOL = ROOT / "tools" / "mirror_git.py"


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def git(*args: str, cwd: Path) -> str:
    result = run("git", *args, cwd=cwd)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class GitMirrorTests(unittest.TestCase):
    def make_remote(self, root: Path) -> tuple[Path, Path]:
        work = root / "work"
        remote = root / "remote.git"
        git("init", "-b", "main", str(work), cwd=root)
        git("config", "user.name", "Gym Test", cwd=work)
        git("config", "user.email", "gym@example.invalid", cwd=work)
        (work / "data.txt").write_text("one\n", encoding="utf-8")
        git("add", "data.txt", cwd=work)
        git("commit", "-m", "first", cwd=work)
        git("tag", "v1", cwd=work)
        git("branch", "feature", cwd=work)
        git("clone", "--bare", str(work), str(remote), cwd=root)
        return work, remote

    def run_tool(
        self, mode: str, mirror_root: Path, manifest: Path
    ) -> subprocess.CompletedProcess[str]:
        return run(
            sys.executable,
            str(MIRROR_TOOL),
            mode,
            str(mirror_root),
            str(manifest),
        )

    def test_update_creates_complete_mirror_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work, remote = self.make_remote(root)
            mirror_root = root / "mirrors"
            manifest = root / "mirrors.tsv"
            manifest.write_text(
                f"local/example.git\t{remote}\n",
                encoding="utf-8",
            )

            first = self.run_tool("update", mirror_root, manifest)
            self.assertEqual(0, first.returncode, first.stderr)
            self.assertIn("PASS mode=update path=local/example.git", first.stdout)

            mirror = mirror_root / "local" / "example.git"
            self.assertEqual("true", git("rev-parse", "--is-bare-repository", cwd=mirror))
            self.assertTrue(git("show-ref", "--verify", "refs/heads/main", cwd=mirror))
            self.assertTrue(git("show-ref", "--verify", "refs/heads/feature", cwd=mirror))
            self.assertTrue(git("show-ref", "--verify", "refs/tags/v1", cwd=mirror))

            (work / "data.txt").write_text("two\n", encoding="utf-8")
            git("add", "data.txt", cwd=work)
            git("commit", "-m", "second", cwd=work)
            git("tag", "v2", cwd=work)
            git("branch", "-D", "feature", cwd=work)
            git("push", "--mirror", str(remote), cwd=work)

            second = self.run_tool("update", mirror_root, manifest)
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertTrue(git("show-ref", "--verify", "refs/tags/v2", cwd=mirror))
            missing = run(
                "git",
                "show-ref",
                "--verify",
                "refs/heads/feature",
                cwd=mirror,
            )
            self.assertNotEqual(0, missing.returncode)

    def test_verify_does_not_contact_missing_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, remote = self.make_remote(root)
            mirror_root = root / "mirrors"
            manifest = root / "mirrors.tsv"
            manifest.write_text(f"repo.git\t{remote}\n", encoding="utf-8")

            update = self.run_tool("update", mirror_root, manifest)
            self.assertEqual(0, update.returncode, update.stderr)
            remote.rename(root / "remote-offline.git")

            verify = self.run_tool("verify", mirror_root, manifest)
            self.assertEqual(0, verify.returncode, verify.stderr)
            self.assertIn("PASS mode=verify path=repo.git", verify.stdout)

    def test_verify_refuses_origin_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, remote = self.make_remote(root)
            mirror_root = root / "mirrors"
            manifest = root / "mirrors.tsv"
            manifest.write_text(f"repo.git\t{remote}\n", encoding="utf-8")

            update = self.run_tool("update", mirror_root, manifest)
            self.assertEqual(0, update.returncode, update.stderr)
            mirror = mirror_root / "repo.git"
            git("remote", "set-url", "origin", str(root / "other.git"), cwd=mirror)

            verify = self.run_tool("verify", mirror_root, manifest)
            self.assertEqual(1, verify.returncode)
            self.assertIn("refusing to retarget", verify.stderr)

    def test_manifest_rejects_parent_escape_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror_root = root / "mirrors"
            manifest = root / "mirrors.tsv"

            manifest.write_text("../escape.git\tlocal\n", encoding="utf-8")
            escape = self.run_tool("verify", mirror_root, manifest)
            self.assertEqual(1, escape.returncode)
            self.assertIn("invalid mirror path", escape.stderr)

            manifest.write_text(
                "a.git\tone\n"
                "a.git\ttwo\n",
                encoding="utf-8",
            )
            duplicate = self.run_tool("verify", mirror_root, manifest)
            self.assertEqual(1, duplicate.returncode)
            self.assertIn("duplicate mirror path", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
