import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    from llm.models.lora_differential import RankOneLoRA
else:
    RankOneLoRA = None


@unittest.skipIf(torch is None, "PyTorch is exercised by the dedicated LoRA workflow")
class RankOneLoRATest(unittest.TestCase):
    def make_adapter(self):
        torch.manual_seed(11)
        base = nn.Linear(3, 2, bias=False)
        base.weight.requires_grad_(False)
        return RankOneLoRA(base, seed=7)

    def test_zero_initialized_b_is_exact_noop(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        expected = adapter.base(x)
        actual = adapter(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.count_nonzero(adapter.b).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.delta_weight()).item(), 0)

    def test_disable_and_zero_merge_are_exact(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        expected = adapter(x)
        adapter.enabled = False
        self.assertTrue(torch.equal(adapter(x), expected))
        adapter.enabled = True
        adapter.merge_()
        self.assertTrue(torch.equal(adapter(x), expected))
        adapter.unmerge_()
        self.assertTrue(torch.equal(adapter(x), expected))

    def test_first_step_moves_b_not_a(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        target = torch.tensor([[0.5, -0.25]])
        optimizer = torch.optim.SGD([adapter.a, adapter.b], lr=0.1)
        a_before = adapter.a.detach().clone()
        b_before = adapter.b.detach().clone()
        base_before = adapter.base.weight.detach().clone()

        loss = (adapter(x) - target).square().sum()
        loss.backward()
        self.assertEqual(torch.count_nonzero(adapter.a.grad).item(), 0)
        self.assertNotEqual(torch.count_nonzero(adapter.b.grad).item(), 0)
        optimizer.step()

        self.assertTrue(torch.equal(adapter.a, a_before))
        self.assertFalse(torch.equal(adapter.b, b_before))
        self.assertTrue(torch.equal(adapter.base.weight, base_before))

    def test_second_step_can_move_a(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        target = torch.tensor([[0.5, -0.25]])
        optimizer = torch.optim.SGD([adapter.a, adapter.b], lr=0.1)

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (adapter(x) - target).square().sum()
            loss.backward()
            if step == 1:
                self.assertNotEqual(torch.count_nonzero(adapter.a.grad).item(), 0)
            optimizer.step()


# Boundary tests use no model downloads or provider calls.
import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock


class RunBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "llm/models/run_pythia_14m_lora_inspectability.py"
        spec = importlib.util.spec_from_file_location("gym_run_boundary", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.args = self.runner.parse_args([])
        self.source = {"commit_sha": "a" * 40, "tracked_tree_clean": True}
        self.model_source = {"repository": self.args.model, "commit_sha": "b" * 40,
                             "requested_revision": self.args.revision, "files": []}
        self.environment = {"python": "fixture", "packages": {"torch": "fixture"}}

    def plan(self):
        return self.runner.build_run_plan(self.args, self.source,
                                           self.model_source, self.environment)

    def test_key_order_does_not_change_identity(self):
        first = self.plan()
        self.environment = dict(reversed(list(self.environment.items())))
        self.assertEqual(first, self.plan())

    def test_revision_alias_does_not_change_identity(self):
        first = self.plan()
        self.args.revision = "another-name-for-same-commit"
        self.model_source["requested_revision"] = self.args.revision
        self.assertEqual(first, self.plan())

    def test_material_coordinates_change_identity(self):
        baseline = self.plan()["job_id"]
        for field, value in {"alpha": 2.0, "learning_rate": 0.01, "seed": 3,
                             "steps": 2, "threads": 2, "example": "Another example.",
                             "target": "gpt_neox.layers.1.attention.query_key_value"}.items():
            with self.subTest(field=field):
                previous = getattr(self.args, field)
                setattr(self.args, field, value)
                self.assertNotEqual(baseline, self.plan()["job_id"])
                setattr(self.args, field, previous)
        for target, key in ((self.source, "commit_sha"), (self.model_source, "commit_sha")):
            previous = target[key]
            target[key] = "c" * 40
            self.assertNotEqual(baseline, self.plan()["job_id"])
            target[key] = previous
        self.environment["python"] = "different-runtime"
        self.assertNotEqual(baseline, self.plan()["job_id"])

    def test_no_plan_is_paid_authority(self):
        plan = self.plan()
        self.assertIs(plan["paid_execution_allowed"], False)
        self.assertNotIn("passed", plan)
        self.assertNotIn("authorization", plan)

    def test_missing_or_stale_plan_fails(self):
        plan = self.plan()
        for expected in (None, {}, {"job_id": plan["job_id"]}, {**plan, "job_id": "0" * 64}):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.runner.require_same_plan(expected, plan)
        stale = copy.deepcopy(plan)
        stale["coordinates"]["model_commit"] = "c" * 40
        with self.assertRaises(ValueError):
            self.runner.require_same_plan(stale, plan)
        self.runner.require_same_plan(copy.deepcopy(plan), plan)

    def test_numeric_false_cannot_replace_boolean_false(self):
        plan = self.plan()
        changed = copy.deepcopy(plan)
        changed["paid_execution_allowed"] = 0
        with self.assertRaises(ValueError):
            self.runner.require_same_plan(changed, plan)

    def test_nonfinite_parameters_fail(self):
        for name in ("alpha", "learning_rate"):
            for value in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    args = copy.deepcopy(self.args)
                    setattr(args, name, value)
                    self.runner.validate_args(args)

    def test_dirty_or_wrong_model_source_fails(self):
        self.source["tracked_tree_clean"] = False
        with self.assertRaises(ValueError):
            self.plan()
        self.source["tracked_tree_clean"] = True
        self.model_source["repository"] = "not/the-requested-model"
        with self.assertRaises(ValueError):
            self.plan()

    def test_mutable_or_missing_commits_fail(self):
        for bad in (None, "main", "step143000", "deadbeef", "z" * 40, "a" * 41):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.runner.require_commit(bad, "model")

    def test_resolver_rejects_host_commit_mismatch(self):
        api = mock.Mock()
        api.model_info.return_value = types.SimpleNamespace(sha="c" * 40, siblings=[])
        module = types.SimpleNamespace(HfApi=mock.Mock(return_value=api))
        with mock.patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaises(ValueError):
                self.runner.hf_source_receipt(self.args.model, "b" * 40)

    def test_model_and_tokenizer_use_one_resolved_commit(self):
        api = mock.Mock()
        api.model_info.side_effect = [
            types.SimpleNamespace(sha="b" * 40, siblings=[]),
            types.SimpleNamespace(sha="c" * 40, siblings=[]),
        ]
        hub = types.SimpleNamespace(HfApi=mock.Mock(return_value=api))
        model_class, tokenizer_class = mock.Mock(), mock.Mock()
        transformers = types.SimpleNamespace(AutoModelForCausalLM=model_class,
                                             AutoTokenizer=tokenizer_class)
        fake_torch = types.SimpleNamespace(float32="fixture-fp32")
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub, "torch": fake_torch,
                                          "transformers": transformers}):
            source = self.runner.hf_source_receipt(self.args.model, "moving-tag")
            self.runner.load_pinned_model(source)
        self.assertEqual(api.model_info.call_count, 1)
        for cls in (model_class, tokenizer_class):
            self.assertEqual(cls.from_pretrained.call_args.kwargs["revision"], "b" * 40)
            self.assertIs(cls.from_pretrained.call_args.kwargs["trust_remote_code"], False)

    def test_actual_git_checkout_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args],
                                               stderr=subprocess.PIPE, text=True).strip()
            git("init")
            git("config", "user.name", "test fixture")
            git("config", "user.email", "fixture@example.invalid")
            path = root / "source.txt"
            path.write_text("original")
            git("add", "source.txt")
            git("commit", "-m", "fixture")
            sha = git("rev-parse", "HEAD")
            with mock.patch.dict(os.environ, {"GYM_SOURCE_SHA": sha}):
                self.assertEqual(self.runner.source_receipt(root)["commit_sha"], sha)
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "uncommitted"):
                    self.runner.source_receipt(root)
                path.write_text("original")
            with mock.patch.dict(os.environ, {"GYM_SOURCE_SHA": "f" * 40}):
                with self.assertRaisesRegex(ValueError, "expected source"):
                    self.runner.source_receipt(root)

    def test_retries_do_not_overwrite_old_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}):
                old = self.runner.reserve_run_directory(root, self.plan()["job_id"])
                (old / "keep").write_text("evidence")
                with self.assertRaises(FileExistsError):
                    self.runner.reserve_run_directory(root, self.plan()["job_id"])
            with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}):
                new = self.runner.reserve_run_directory(root, self.plan()["job_id"])
            self.assertNotEqual(old, new)
            self.assertEqual((old / "keep").read_text(), "evidence")

    def test_bad_run_identifiers_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for values in ({"GITHUB_RUN_ID": "../x", "GITHUB_RUN_ATTEMPT": "1"},
                           {"GITHUB_RUN_ID": "2"}, {"GITHUB_RUN_ATTEMPT": "1"}):
                with mock.patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                    self.runner.reserve_run_directory(Path(tmp), self.plan()["job_id"])

    def test_atomic_json_rejects_nan_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            self.runner.atomic_json(path, {"state": "original"})
            with self.assertRaises(ValueError):
                self.runner.atomic_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"state": "original"})

    def test_artifact_index_detects_changed_missing_and_added_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "tensor.pt"
            payload.write_bytes(b"abcd")
            self.runner.write_artifact_index(root)
            self.runner.verify_artifact_index(root)
            payload.write_bytes(b"abce")
            with self.assertRaises(ValueError):
                self.runner.verify_artifact_index(root)
            payload.unlink()
            with self.assertRaises(ValueError):
                self.runner.verify_artifact_index(root)
            payload.write_bytes(b"abcd")
            (root / "unexpected").write_bytes(b"x")
            with self.assertRaises(ValueError):
                self.runner.verify_artifact_index(root)

    def test_artifact_index_rejects_duplicates_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").write_bytes(b"x")
            self.runner.write_artifact_index(root)
            path = root / "artifact-index.json"
            original = json.loads(path.read_text())
            duplicate = copy.deepcopy(original)
            duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
            path.write_text(json.dumps(duplicate))
            with self.assertRaises(ValueError):
                self.runner.verify_artifact_index(root)
            original["files"][0]["path"] = "../outside"
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                self.runner.verify_artifact_index(root)

    def test_artifact_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").write_bytes(b"x")
            (root / "alias").symlink_to(root / "data")
            with self.assertRaises(ValueError):
                self.runner.write_artifact_index(root)

    def invoke_main(self, root, extra=(), execute=None):
        with mock.patch.object(self.runner, "source_receipt", return_value=self.source), \
             mock.patch.object(self.runner, "hf_source_receipt", return_value=self.model_source), \
             mock.patch.object(self.runner.metadata, "version", return_value="fixture"), \
             mock.patch.object(self.runner, "execute_plan", side_effect=execute) as engine, \
             mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = self.runner.main(["--output-dir", str(root), *extra])
        return result, engine

    def test_plan_only_does_not_execute_or_claim_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, engine = self.invoke_main(Path(tmp), ["--plan-only"])
            self.assertEqual(result, 0)
            engine.assert_not_called()
            directory, = Path(tmp).iterdir()
            self.assertTrue((directory / "plan.json").is_file())
            self.assertFalse((directory / "COMPLETE.json").exists())
            self.assertFalse((directory / "manifest.json").exists())

    def test_mismatched_plan_stops_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong = root / "expected.json"
            wrong.write_text("{}")
            with self.assertRaisesRegex(ValueError, "different job"):
                self.invoke_main(root / "outputs", ["--expected-plan", str(wrong)])
            self.assertFalse((root / "outputs").exists())

    def test_failed_execution_retains_failure_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail(args, directory, source, model_source, plan):
                (directory / "partial.pt").write_bytes(b"partial")
                raise RuntimeError("injected failure")
            with self.assertRaises(RuntimeError):
                self.invoke_main(Path(tmp), execute=fail)
            directory, = Path(tmp).iterdir()
            self.assertTrue((directory / "partial.pt").exists())
            self.assertFalse((directory / "COMPLETE.json").exists())
            self.assertIs(json.loads((directory / "FAILED.json").read_text())["passed"], False)

    def test_false_acceptance_cannot_create_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.invoke_main(Path(tmp), execute=lambda *args: {"passed": False})
            directory, = Path(tmp).iterdir()
            self.assertFalse((directory / "COMPLETE.json").exists())

    def test_success_marker_binds_verified_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            def finish(args, directory, source, model_source, plan):
                self.runner.atomic_json(directory / "manifest.json", {"passed": True})
                (directory / "tensor.pt").write_bytes(b"synthetic fixture, not model evidence")
                return {"passed": True}
            result, _ = self.invoke_main(Path(tmp), execute=finish)
            self.assertEqual(result, 0)
            directory, = Path(tmp).iterdir()
            complete = json.loads((directory / "COMPLETE.json").read_text())
            self.assertEqual(complete["artifact_index_sha256"],
                             self.runner.file_sha256(directory / "artifact-index.json"))
            self.assertIs(complete["paid_execution_allowed"], False)
            self.runner.verify_artifact_index(directory)

    def exercise_execute_plan_failure(self, test0, steps):
        class Tokens:
            shape = (1, 2)
            def clone(self):
                return self
            def tolist(self):
                return [[1, 2]]
        fake_torch = types.SimpleNamespace(float32="fixture-fp32",
                                           set_num_threads=mock.Mock(), manual_seed=mock.Mock())
        lora = types.SimpleNamespace(
            attach_lora=mock.Mock(), run_test0=mock.Mock(return_value=test0),
            run_steps=mock.Mock(return_value=steps), write_run_manifest=mock.Mock())
        tokenizer = mock.Mock(return_value={"input_ids": Tokens()})
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(sys.modules, {"torch": fake_torch,
                 "huggingface_hub": types.SimpleNamespace(__version__="fixture"),
                 "lora_differential": lora}), \
             mock.patch.object(self.runner, "load_pinned_model", return_value=(mock.Mock(), tokenizer)):
            with self.assertRaises(ValueError):
                self.runner.execute_plan(self.args, Path(tmp), self.source,
                                         self.model_source, self.plan())
        return lora

    def test_failed_zero_update_stops_before_optimizer(self):
        lora = self.exercise_execute_plan_failure({"passed": False}, [])
        lora.run_steps.assert_not_called()
        lora.write_run_manifest.assert_not_called()

    def test_incomplete_optimizer_receipts_cannot_pass(self):
        self.args.steps = 2
        lora = self.exercise_execute_plan_failure({"passed": True}, [{"passed": True}])
        lora.run_steps.assert_called_once()
        lora.write_run_manifest.assert_not_called()

    def test_workflow_checks_out_exact_head_in_both_jobs(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/pythia-14m-lora-inspectability.yml"
        text = path.read_text()
        self.assertEqual(text.count("ref: ${{ github.event.pull_request.head.sha || github.sha }}"), 2)
        self.assertNotIn('steps="${{ inputs.steps }}"', text)
        self.assertIn('steps="$REQUESTED_STEPS"', text)
        self.assertNotIn("    paths:", text)


if __name__ == "__main__":
    unittest.main()
