"""Offline acceptance: synthetic identities are fixtures, not real model evidence."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import experiment_contract as ec

NOW = 10000


def spec():
    return {
        "schema": 1,
        "source": {"repository": "fixture/gym", "commit": "a" * 40, "dirty": False},
        "environment_sha256": "b" * 64,
        "input_sha256": "c" * 64,
        "required_checks": ["zero_update", "one_step"],
        "fixed": {"model": {"repository": "fixture/14m", "revision": "d" * 40},
                  "adapter": {"rank": 1, "alpha": "1"},
                  "example": "The moon orbits the Earth.", "optimizer": "SGD", "steps": 2},
        "axes": {"adapter.target": ["layer0.qkv", "layer0.mlp"], "seed": [0, 1]},
        "exclude": [], "max_cases": 8,
    }


def case_and_job():
    case = ec.expand(spec())[0]
    job = ec.make_job(case, "vast", {"ram_bytes": 1000, "disk_bytes": 2000,
            "runtime_seconds": 300, "accelerator": "none", "accelerator_count": 0},
            {"offer_id": "synthetic-offer", "image_digest": "sha256:" + "e" * 64})
    return case, job


def evidence_event(case, job, stage, payload, at):
    return {"stage": stage, "status": "pass", "job_id": job["job_id"],
            "case_id": case["case_id"], "source_commit": case["source"]["commit"],
            "at": at, "evidence": payload,
            "evidence_sha256": ec.digest("gym-evidence-v1", payload)}


def trace(case, job):
    request = ec.digest("gym-request-v1", job["request"])
    payloads = [
        {"source": copy.deepcopy(case["source"])},
        {"environment_sha256": case["environment_sha256"], "compatible": True},
        {"executor": "github-standard", "run_id": 1, "run_attempt": 1, "coverage": "exact",
         "environment_sha256": case["environment_sha256"], "result_sha256": "f" * 64,
         "checks": {name: "pass" for name in case["required_checks"]}},
        {"basis": "measured", "measurement_sha256": "1" * 64,
         "ram_bytes": 500, "disk_bytes": 1000, "runtime_seconds": 150},
        {"dry_run": True, "mutations": 0, "request_sha256": request,
         "currency": "USD", "total_microunits": 100000, "expires_at": NOW + 300},
    ]
    events = [evidence_event(case, job, stage, payload, NOW - 10 + i)
              for i, (stage, payload) in enumerate(zip(ec.STAGES, payloads))]
    approval = {"approved": True, "operator": "fixture-only", "request_sha256": request,
                "preflight_sha256": ec.digest("gym-preflight-v1", events), "currency": "USD",
                "max_total_microunits": 200000, "expires_at": NOW + 300}
    events.append(evidence_event(case, job, ec.STAGES[-1], approval, NOW - 5))
    return events


def reseal(events):
    # Simulate an internally consistent but semantically invalid fixture. Hash
    # verification alone must not be the reason every negative test fails.
    for event in events:
        event["evidence_sha256"] = ec.digest("gym-evidence-v1", event["evidence"])
    events[-1]["evidence"]["preflight_sha256"] = ec.digest("gym-preflight-v1", events[:-1])
    events[-1]["evidence_sha256"] = ec.digest("gym-evidence-v1", events[-1]["evidence"])


class MatrixTests(unittest.TestCase):
    def test_nested_product_keeps_all_coordinates(self):
        cases = ec.expand(spec())
        self.assertEqual(len(cases), 4)
        self.assertEqual({(x["coordinates"]["adapter"]["target"], x["coordinates"]["seed"]) for x in cases},
                         {(a, b) for a in ["layer0.qkv", "layer0.mlp"] for b in [0, 1]})
        self.assertTrue(all(x["coordinates"]["steps"] == 2 for x in cases))

    def test_axis_order_value_order_and_duplicates_do_not_change_ids(self):
        a = spec()
        b = spec()
        b["axes"] = {"seed": [1, 0, 1], "adapter.target": ["layer0.mlp", "layer0.qkv"]}
        self.assertEqual(ec.expand(a), ec.expand(b))

    def test_one_example_is_not_replaced_with_a_batch(self):
        for case in ec.expand(spec()):
            self.assertEqual(case["coordinates"]["example"], "The moon orbits the Earth.")
            self.assertEqual(case["coordinates"]["adapter"]["rank"], 1)

    def test_grouped_models_keep_repository_and_revision_together(self):
        matrix = spec()
        del matrix["fixed"]["model"]
        matrix["axes"] = {"model": [{"repository": "fixture/a", "revision": "a" * 40},
                                     {"repository": "fixture/b", "revision": "b" * 40}]}
        matrix["exclude"] = [{"model.repository": "fixture/a"}]
        cases = ec.expand(matrix)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["coordinates"]["model"]["repository"], "fixture/b")

    def test_empty_axes_mean_one_fixed_case(self):
        matrix = spec()
        matrix["axes"] = {}
        self.assertEqual(len(ec.expand(matrix)), 1)

    def test_case_identity_binds_sources_environment_input_and_parameters(self):
        original = {x["case_id"] for x in ec.expand(spec())}
        for field in ["source", "environment_sha256", "input_sha256", "required_checks", "model", "alpha"]:
            with self.subTest(field=field):
                matrix = spec()
                if field == "source": matrix[field]["commit"] = "f" * 40
                elif field in ("environment_sha256", "input_sha256"): matrix[field] = "f" * 64
                elif field == "required_checks": matrix[field].append("reload")
                elif field == "model": matrix["fixed"][field]["revision"] = "f" * 40
                else: matrix["fixed"]["adapter"][field] = "2"
                self.assertTrue(original.isdisjoint(x["case_id"] for x in ec.expand(matrix)))

    def test_exclusion_is_exact_and_typed(self):
        matrix = spec()
        matrix["exclude"] = [{"adapter.target": "layer0.qkv", "seed": 1}]
        self.assertEqual(len(ec.expand(matrix)), 3)
        matrix["exclude"] = [{"seed": True}]
        with self.assertRaisesRegex(ec.ContractError, "EXCLUSION-MATCHES-NOTHING"):
            ec.expand(matrix)

    def test_no_silent_empty_or_oversized_or_misspelled_matrix(self):
        for kind in ["empty", "big", "typo", "all_excluded", "unused_exclusion"]:
            with self.subTest(kind=kind):
                matrix = spec()
                if kind == "empty": matrix["axes"]["seed"] = []
                elif kind == "big": matrix["axes"]["seed"] = list(range(100))
                elif kind == "typo": matrix["exclude"] = [{"sead": 1}]
                elif kind == "all_excluded": matrix["exclude"] = [{"seed": 0}, {"seed": 1}]
                else: matrix["exclude"] = [{"seed": 999}]
                with self.assertRaises(ec.ContractError): ec.expand(matrix)

    def test_fixed_axis_and_ancestor_conflicts_rejected(self):
        for axis, values in [("adapter.rank", [2]), ("adapter", [{}])]:
            matrix = spec()
            matrix["axes"][axis] = values
            with self.assertRaisesRegex(ec.ContractError, "AXIS-CONFLICT"): ec.expand(matrix)

    def test_floats_nan_dirty_source_and_mutable_model_fail(self):
        for kind in ["float", "nan", "dirty", "model_revision", "source_revision", "boolean_schema"]:
            with self.subTest(kind=kind):
                matrix = spec()
                if kind == "float": matrix["fixed"]["lr"] = 0.1
                elif kind == "nan": matrix["fixed"]["lr"] = float("nan")
                elif kind == "dirty": matrix["source"]["dirty"] = True
                elif kind == "model_revision": matrix["fixed"]["model"]["revision"] = "step143000"
                elif kind == "source_revision": matrix["source"]["commit"] = "main"
                else: matrix["schema"] = True
                with self.assertRaises(ec.ContractError): ec.expand(matrix)

    def test_hash_collision_is_not_silent_deduplication(self):
        with patch.object(ec, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(ec.ContractError, "ID-COLLISION"): ec.expand(spec())

    def test_strict_json_rejects_duplicate_and_float_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ['{"seed":0,"seed":1}', '{"lr":0.1}', '{"lr":NaN}']:
                path.write_text(text)
                with self.assertRaises(ec.ContractError): ec.load_json(path)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.case, self.job = case_and_job()
        self.events = trace(self.case, self.job)

    def check(self, events=None, job=None):
        return ec.audit_preflight(self.case, job or self.job, events or self.events, now=NOW)

    def test_valid_trace_does_not_enable_paid_execution(self):
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("process forbidden")):
            result = self.check()
        self.assertTrue(result["contract_valid"])
        self.assertFalse(result["execution_enabled"])

    def test_gpu_request_cannot_be_approved_by_cpu_smoke(self):
        resources = {**self.job["resources"], "accelerator": "A100", "accelerator_count": 1}
        job = ec.make_job(self.case, "vast", resources, self.job["request"])
        events = trace(self.case, job)
        with self.assertRaisesRegex(ec.ContractError, "GPU-PREFLIGHT-NOT-IMPLEMENTED"):
            self.check(events, job)

    def test_contract_module_has_no_network_or_process_imports(self):
        tree = ast.parse((ROOT / "tools/experiment_contract.py").read_text())
        allowed = {"__future__", "argparse", "copy", "hashlib", "itertools", "json", "re",
                   "stat", "sys", "time", "pathlib", "typing"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(item.name in allowed for item in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"eval", "exec", "__import__"})

    def test_all_missing_stages_and_reordered_events_rejected(self):
        for i in range(len(self.events)):
            events = copy.deepcopy(self.events)
            del events[i]
            with self.assertRaisesRegex(ec.ContractError, "TRACE-INCOMPLETE"): self.check(events)
        events = copy.deepcopy(self.events)
        events[2], events[3] = events[3], events[2]
        with self.assertRaisesRegex(ec.ContractError, "STAGE-ORDER"): self.check(events)

    def test_failure_skip_cancel_unknown_cannot_count_as_success(self):
        for i in range(len(self.events)):
            for status in ["fail", "skipped", "cancelled", "unknown"]:
                with self.subTest(stage=i, status=status):
                    events = copy.deepcopy(self.events)
                    events[i]["status"] = status
                    with self.assertRaisesRegex(ec.ContractError, "STAGE-NOT-PASSED"): self.check(events)

    def test_stale_future_backwards_and_wrong_source_or_job_fail(self):
        mutations = [("at", NOW - ec.MAX_AGE_SECONDS - 1), ("at", NOW + 1),
                     ("at", NOW - 100), ("source_commit", "0" * 40), ("job_id", "0" * 64)]
        for key, value in mutations:
            events = copy.deepcopy(self.events)
            events[2][key] = value
            with self.assertRaises(ec.ContractError): self.check(events)

    def test_changed_job_request_or_resource_invalidates_evidence(self):
        for field in ["request", "resources", "provider"]:
            body = {k: copy.deepcopy(v) for k, v in self.job.items() if k != "job_id"}
            if field == "request": body[field]["offer_id"] = "new-offer"
            elif field == "resources": body[field]["ram_bytes"] += 1
            else: body[field] = "hetzner"
            new_job = ec.seal("job", body)
            self.assertEqual(new_job["case_id"], self.job["case_id"])
            self.assertNotEqual(new_job["job_id"], self.job["job_id"])
            with self.assertRaisesRegex(ec.ContractError, "WRONG-JOB"): self.check(job=new_job)

    def test_semantic_failures_survive_rehashing(self):
        variants = [
            (0, "source", {**self.case["source"], "dirty": True}, "STALE-SOURCE"),
            (1, "compatible", False, "INCOMPATIBLE"),
            (2, "coverage", "surrogate", "SURROGATE-NOT-AUTHORITY"),
            (2, "run_attempt", 0, "SMOKE-ATTEMPT"),
            (2, "checks", {"zero_update": "pass"}, "CHECK-COVERAGE"),
            (2, "checks", {"zero_update": "pass", "one_step": "skipped"}, "CHECK-NOT-PASSED"),
            (2, "environment_sha256", "0" * 64, "SMOKE-ENVIRONMENT"),
            (3, "basis", "guessed", "UNMEASURED-RESOURCES"),
            (3, "ram_bytes", 99999, "RESOURCE-ENVELOPE"),
            (4, "dry_run", False, "NOT-DRY-RUN"),
            (4, "mutations", 1, "NOT-DRY-RUN"),
            (4, "request_sha256", "0" * 64, "REQUEST-MISMATCH"),
            (4, "expires_at", NOW, "QUOTE-EXPIRED"),
            (4, "total_microunits", 999999, "OVER-BUDGET"),
            (5, "approved", False, "NO-APPROVAL"),
            (5, "operator", "  ", "NO-APPROVAL"),
            (5, "currency", "EUR", "CURRENCY-MISMATCH"),
            (5, "max_total_microunits", -1, "BUDGET"),
            (5, "max_total_microunits", 0, "OVER-BUDGET"),
            (5, "expires_at", NOW, "APPROVAL-EXPIRED"),
        ]
        for index, field, value, code in variants:
            with self.subTest(stage=index, field=field):
                events = copy.deepcopy(self.events)
                events[index]["evidence"][field] = value
                reseal(events)
                with self.assertRaisesRegex(ec.ContractError, code): self.check(events)

    def test_approval_is_bound_to_exact_prior_trace(self):
        events = copy.deepcopy(self.events)
        events[4]["evidence"]["total_microunits"] -= 1
        events[4]["evidence_sha256"] = ec.digest("gym-evidence-v1", events[4]["evidence"])
        with self.assertRaisesRegex(ec.ContractError, "APPROVAL-BINDING"): self.check(events)

    def test_tampered_evidence_and_boolean_numeric_fields_rejected(self):
        self.events[2]["evidence"]["run_id"] = 5
        with self.assertRaisesRegex(ec.ContractError, "EVIDENCE-DIGEST"): self.check()
        self.events = trace(self.case, self.job)
        self.events[2]["evidence"]["run_id"] = True
        reseal(self.events)
        with self.assertRaisesRegex(ec.ContractError, "SMOKE-RUN"): self.check()


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case, self.job = case_and_job()
        payload = b"synthetic tensor fixture"
        (self.root / "step-0001.bin").write_bytes(payload)
        self.receipt = {"schema": 1, "job_id": self.job["job_id"], "case_id": self.case["case_id"],
                        "status": "complete", "checks": {"zero_update": "pass", "one_step": "pass"},
                        "artifacts": [{"path": "step-0001.bin", "size_bytes": len(payload),
                                       "sha256": hashlib.sha256(payload).hexdigest()}]}
        self.write()

    def write(self):
        (self.root / "run.json").write_text(json.dumps(self.receipt))

    def verify(self):
        return ec.verify_run(self.root, self.case, self.job)

    def test_independent_readback(self):
        self.assertTrue(self.verify()["verified"])

    def test_corrupt_missing_or_unindexed_bytes_rejected(self):
        (self.root / "step-0001.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-CORRUPT"): self.verify()
        (self.root / "step-0001.bin").unlink()
        with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-MISSING"): self.verify()

    def test_extra_file_rejected(self):
        (self.root / "unindexed.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ec.ContractError, "UNINDEXED-ARTIFACT"): self.verify()

    def test_interrupted_run_without_final_receipt_is_not_success(self):
        (self.root / "run.json").unlink()
        with self.assertRaisesRegex(ec.ContractError, "MISSING-RECEIPT"): self.verify()
        self.receipt["status"] = "running"
        self.write()
        with self.assertRaisesRegex(ec.ContractError, "INCOMPLETE-RUN"): self.verify()

    def test_wrong_job_skipped_check_empty_bundle_and_bad_size_rejected(self):
        for field, value in [("job_id", "0" * 64), ("checks", {"zero_update": "pass", "one_step": "skipped"}),
                             ("artifacts", [])]:
            saved = copy.deepcopy(self.receipt)
            self.receipt[field] = value
            self.write()
            with self.assertRaises(ec.ContractError): self.verify()
            self.receipt = saved
        self.receipt["artifacts"][0]["size_bytes"] += 1
        self.write()
        with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-CORRUPT"): self.verify()

    def test_path_traversal_duplicate_and_symlink_rejected(self):
        original = self.receipt["artifacts"][0]["path"]
        for name in ["../outside", "/outside", "a//b", "./step-0001.bin", "run.json", "C:\\file"]:
            self.receipt["artifacts"][0]["path"] = name
            self.write()
            with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-PATH"): self.verify()
        self.receipt["artifacts"][0]["path"] = original
        self.receipt["artifacts"].append(copy.deepcopy(self.receipt["artifacts"][0]))
        self.write()
        with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-PATH"): self.verify()
        self.receipt["artifacts"].pop()
        self.write()
        (self.root / original).unlink()
        (self.root / original).symlink_to(self.root / "run.json")
        with self.assertRaisesRegex(ec.ContractError, "ARTIFACT-SYMLINK"): self.verify()

    def test_cli_expand_and_invalid_json(self):
        matrix = self.root / "matrix.json"
        matrix.write_text(json.dumps(spec()))
        run = subprocess.run([sys.executable, str(ROOT / "tools/experiment_contract.py"), "expand", str(matrix)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(json.loads(run.stdout)["cases"]), 4)
        matrix.write_text('{"x":1,"x":2}')
        run = subprocess.run([sys.executable, str(ROOT / "tools/experiment_contract.py"), "expand", str(matrix)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertFalse(json.loads(run.stderr)["execution_enabled"])


if __name__ == "__main__":
    unittest.main()
