#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest

import run_ollama_context as context_runner


def git(repo: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.decode("utf-8").strip()


class FrozenContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.temporary.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Context Test")
        git(self.repo, "config", "user.email", "context@example.invalid")
        (self.repo / "a.md").write_text("alpha from commit\n", encoding="utf-8")
        (self.repo / "b.md").write_text("beta from commit\n", encoding="utf-8")
        git(self.repo, "add", "a.md", "b.md")
        git(self.repo, "commit", "-q", "-m", "fixture")
        self.commit = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_context_reads_commit_objects_not_working_tree(self) -> None:
        first = context_runner.build_context(
            self.repo,
            self.commit,
            ["b.md", "a.md", "a.md"],
        )

        (self.repo / "a.md").write_text("uncommitted replacement\n", encoding="utf-8")
        second = context_runner.build_context(
            self.repo,
            self.commit,
            ["a.md", "b.md"],
        )

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(
            [entry["path"] for entry in first["files"]],
            ["a.md", "b.md"],
        )
        self.assertIn("alpha from commit", second["text"])
        self.assertNotIn("uncommitted replacement", second["text"])

    def test_conditions_do_not_reuse_the_url_intervention(self) -> None:
        context = context_runner.build_context(self.repo, self.commit, ["a.md"])
        question = "Should I borrow this much for college?"

        without = context_runner.effective_prompt(
            question,
            "without-blackball",
            context["text"],
        )
        with_context = context_runner.effective_prompt(
            question,
            "blackball-context",
            context["text"],
        )

        self.assertEqual(without, question)
        self.assertTrue(with_context.endswith("\n" + question))
        self.assertIn(f"commit: {self.commit}", with_context)
        self.assertIn("alpha from commit", with_context)
        self.assertNotIn("https://github.com/bl4ckb4ll/blackball", with_context)

    def test_symbolic_ref_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact 40-hex"):
            context_runner.freeze_commit(self.repo, "HEAD")


if __name__ == "__main__":
    unittest.main()
