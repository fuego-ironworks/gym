from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_record import validate  # noqa: E402

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def load_schema(name: str) -> dict:
    with (ROOT / "schemas" / name).open("r", encoding="utf-8") as source:
        return json.load(source)


MODEL_MANIFEST = {
    "schema_version": "gym.model-manifest.v1",
    "manifest_id": "example-model-q4",
    "model_name": "Example Model",
    "family": "example",
    "source": {
        "provider": "example-registry",
        "locator": "models/example",
        "revision": "0123456789abcdef",
        "acquired_at": "2026-09-17T16:00:00Z",
    },
    "objects": [
        {"role": "weights", "sha256": HASH_A, "size_bytes": 1234, "format": "gguf"},
        {"role": "tokenizer", "sha256": HASH_B},
    ],
    "runtime": {
        "name": "example-runtime",
        "revision": "runtime-rev-1",
        "requirements": ["cpu"],
    },
    "license": {
        "name": "example-license",
        "source": "retained-license-text",
        "redistribution": "restricted",
    },
}

RUN = {
    "schema_version": "gym.run.v1",
    "run_id": "run-20260917-001",
    "created_at": "2026-09-17T16:05:00Z",
    "work_item": {
        "id": "case-repository-review-001",
        "input_sha256": HASH_C,
        "repository_revision": "deadbeef",
    },
    "model_manifest": {"id": "example-model-q4", "sha256": HASH_D},
    "runner": {"id": "local-shell", "revision": "1", "definition_sha256": HASH_A},
    "runtime": {"name": "example-runtime", "revision": "runtime-rev-1"},
    "scorers": [
        {"id": "exact-output", "revision": "1", "definition_sha256": HASH_B}
    ],
    "policies": [
        {"id": "continuation", "revision": "1", "definition_sha256": HASH_C}
    ],
    "settings": {"temperature": 0},
}

RECEIPT = {
    "schema_version": "gym.receipt.v1",
    "receipt_id": "receipt-20260917-001",
    "run": {"id": "run-20260917-001", "sha256": HASH_A},
    "completed_at": "2026-09-17T16:06:00Z",
    "status": "PASS",
    "checks": [
        {"name": "exact-output", "status": "PASS", "evidence_sha256": HASH_B}
    ],
    "continuations": 0,
    "output_objects": [{"role": "model-output", "sha256": HASH_C}],
    "trace_objects": [{"role": "tool-trace", "sha256": HASH_D, "visibility": "private"}],
}


class SchemaValidationTests(unittest.TestCase):
    def test_valid_records(self) -> None:
        cases = [
            ("model-manifest.schema.json", MODEL_MANIFEST),
            ("run.schema.json", RUN),
            ("receipt.schema.json", RECEIPT),
        ]
        for schema_name, record in cases:
            with self.subTest(schema=schema_name):
                self.assertEqual([], validate(load_schema(schema_name), record))

    def test_bad_hash_is_rejected(self) -> None:
        record = copy.deepcopy(MODEL_MANIFEST)
        record["objects"][0]["sha256"] = "not-a-hash"
        errors = validate(load_schema("model-manifest.schema.json"), record)
        self.assertTrue(any("does not match" in error for error in errors))

    def test_unknown_field_is_rejected(self) -> None:
        record = copy.deepcopy(RECEIPT)
        record["provider_receipt_url"] = "https://example.invalid/receipt/1"
        errors = validate(load_schema("receipt.schema.json"), record)
        self.assertTrue(any("unexpected property" in error for error in errors))

    def test_unsupported_schema_keyword_fails_closed(self) -> None:
        cases = [
            ({"type": "string", "maxLength": 3}, "ok"),
            (
                {
                    "type": "object",
                    "properties": {
                        "optional": {"type": "string", "maxLength": 3},
                    },
                },
                {},
            ),
        ]
        for schema, record in cases:
            with self.subTest(schema=schema):
                errors = validate(schema, record)
                self.assertTrue(
                    any("unsupported schema keyword" in error for error in errors)
                )

    def test_unknown_status_is_preserved_as_a_valid_status(self) -> None:
        record = copy.deepcopy(RECEIPT)
        record["status"] = "UNKNOWN"
        record["checks"][0]["status"] = "UNKNOWN"
        self.assertEqual([], validate(load_schema("receipt.schema.json"), record))

    def test_cli_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record_path = Path(directory) / "record.json"
            record_path.write_text(json.dumps(RUN), encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "tools" / "validate_record.py"),
                str(ROOT / "schemas" / "run.schema.json"),
                str(record_path),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
