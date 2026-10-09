from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from functorial_c_eval import (
    blob_hash, check_answer, check_patch, equal_answer,
    make_prompt, read_manifest, read_sources,
)


class FunctorialCExperimentTests(unittest.TestCase):
    def test_real_pairs_are_pinned_and_matched(self):
        manifest = read_manifest()
        self.assertEqual({p["id"] for p in manifest["pairs"]},
                         {"fourier-horner", "seifert-ribbons"})
        for p in manifest["pairs"]:
            self.assertTrue(p["questions"])
            self.assertNotEqual(p["before"]["commit"], p["after"]["commit"])
            self.assertEqual(
                {f["alias"] for f in p["before"]["files"]},
                {f["alias"] for f in p["after"]["files"]},
            )

    def test_source_cache_sha_verification(self):
        payload = b"int f(void) { return 2; }\n"
        pair = {
            "id": "dummy", "repository": "faux/test",
            "before": {"commit": "a" * 40, "files": [
                {"alias": "code.c", "path": "f.c", "blob_sha": blob_hash(payload)}]}
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            dest = cache / "dummy" / "before" / "code.c"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(payload)
            self.assertEqual(read_sources(pair, "before", cache, True)["code.c"],
                             payload.decode())
            dest.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "blob mismatch"):
                read_sources(pair, "before", cache, True)

    def test_json_answers_are_strict(self):
        self.assertTrue(check_answer('{"answer": {"real":2,"imaginary":5}}',
                                     {"imaginary": 5, "real": 2})["correct"])
        self.assertFalse(check_answer('{"answer":true}', 1)["correct"])
        self.assertFalse(check_answer('{"answer":2,"explanation":"guess"}',
                                      2)["correct"])
        self.assertFalse(check_answer('The answer is 2', 2)["correct"])

    def test_only_single_file_applicable_patch_counts(self):
        original = "int number = 1;\n"
        good = "--- a/code.c\n+++ b/code.c\n@@ -1 +1 @@\n-int number = 1;\n+int number = 2;\n"
        result = check_patch(good, original)
        self.assertTrue(result["applies"], result)
        self.assertEqual(result["semantic_correctness"], "NOT_CHECKED")
        bad = good.replace("b/code.c", "b/secrets")
        self.assertFalse(check_patch(bad, original)["applies"])

    def test_prompt_wrapper_hides_condition(self):
        sources = {"code.c": "int f(void) { return 2; }\n"}
        text = make_prompt(sources, {"question": "What does f return?"}, "question")
        self.assertNotIn("functorial", text.lower())
        self.assertNotIn("before", text.lower())
        self.assertNotIn("after", text.lower())
        self.assertIn('{"answer": VALUE}', text)


if __name__ == "__main__":
    unittest.main()
