from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from object_store import (  # noqa: E402
    CorruptObjectError,
    object_path,
    put_file,
    verify_object,
)


class ObjectStoreTests(unittest.TestCase):
    def test_put_uses_digest_path_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            store = work / "objects"
            source = work / "model.bin"
            payload = b"model-bytes\n"
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            sha256, size, path, created = put_file(store, source)
            self.assertEqual(expected, sha256)
            self.assertEqual(len(payload), size)
            self.assertEqual(object_path(store, expected), path)
            self.assertEqual(payload, path.read_bytes())
            self.assertTrue(created)

            sha256_2, size_2, path_2, created_2 = put_file(store, source)
            self.assertEqual((sha256, size, path), (sha256_2, size_2, path_2))
            self.assertFalse(created_2)

    def test_verify_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            store = work / "objects"
            source = work / "object.bin"
            source.write_bytes(b"good")
            sha256, _, path, _ = put_file(store, source)

            path.write_bytes(b"bad")
            ok, message = verify_object(store, sha256)
            self.assertFalse(ok)
            self.assertIn("CORRUPT", message)

            with self.assertRaises(CorruptObjectError):
                put_file(store, source)

    def test_verify_detects_missing_and_wrong_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            store = work / "objects"
            missing = "0" * 64
            ok, message = verify_object(store, missing)
            self.assertFalse(ok)
            self.assertIn("MISSING", message)

            source = work / "object.bin"
            source.write_bytes(b"1234")
            sha256, _, _, _ = put_file(store, source)
            ok, message = verify_object(store, sha256, expected_size=5)
            self.assertFalse(ok)
            self.assertIn("SIZE_MISMATCH", message)

    def test_cli_put_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            store = work / "objects"
            source = work / "object.bin"
            source.write_bytes(b"cli-data")
            command = [
                sys.executable,
                str(ROOT / "tools" / "object_store.py"),
                "put",
                str(store),
                str(source),
            ]
            put = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(0, put.returncode, put.stderr)
            self.assertIn("STORED", put.stdout)

            sha256 = hashlib.sha256(b"cli-data").hexdigest()
            verify = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "object_store.py"),
                    "verify",
                    str(store),
                    sha256,
                    "--size-bytes",
                    "8",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, verify.returncode, verify.stderr)
            self.assertIn("PASS", verify.stdout)

    def test_cli_rejects_bad_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "object_store.py"),
                    "verify",
                    directory,
                    "NOT-A-DIGEST",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(1, completed.returncode)
            self.assertIn("64 lowercase hexadecimal", completed.stderr)


if __name__ == "__main__":
    unittest.main()
