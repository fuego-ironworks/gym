from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "llm" / "models" / "reproduction" / "pythia-410m-deduped"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeHub:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.hashes: dict[str, str] = {}

    def upload(self, local_path: Path, remote_path: str, message: str) -> str:
        payload = local_path.read_bytes()
        self.uploads.append((remote_path, payload))
        if remote_path.endswith(".tar"):
            self.hashes[remote_path] = hashlib.sha256(payload).hexdigest()
        return "fake-commit"

    def remote_lfs_sha256(self, remote_path: str):
        return self.hashes.get(remote_path)


class PythiaVastNonBillingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retire = load_module("checkpoint_retire", REPRO / "checkpoint_retire.py")

    def test_vast_create_is_dry_run_without_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            marker = work / "vastai-called"
            fake = work / "vastai"
            fake.write_text(
                "#!/bin/sh\necho called > \"$VAST_MARKER\"\nexit 91\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = os.environ.copy()
            env.pop("VAST_CREATE", None)
            env["PATH"] = f"{work}:{env.get('PATH', '')}"
            env["VAST_MARKER"] = str(marker)
            run = subprocess.run(
                ["bash", str(REPRO / "vast_create.sh"), "12345"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertIn("dry run only", run.stdout)
            self.assertFalse(marker.exists(), "dry run invoked Vast CLI")

    def test_vast_opt_in_cannot_bypass_missing_gpu_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            log = work / "args"
            fake = work / "vastai"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$VAST_LOG\"\n", encoding="utf-8")
            fake.chmod(0o755)
            env = os.environ.copy()
            env["VAST_CREATE"] = "1"
            env["PATH"] = f"{work}:{env.get('PATH', '')}"
            env["VAST_LOG"] = str(log)
            run = subprocess.run(
                ["bash", str(REPRO / "vast_create.sh"), "12345"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, run.returncode)
            self.assertIn('BLOCKED',run.stderr)
            self.assertFalse(log.exists())

    def test_dense_schedule_covers_interval_without_early_overretention(self):
        config=load_module('dense_config',REPRO/'make_dense_config.py')
        schedule=config.checkpoint_schedule(8)
        self.assertEqual([512+8*i for i in range(62)], [x for x in schedule if x>=512])
        self.assertEqual(72,len(schedule))
        derived=config.build_config(ROOT/'llm/models/upstream/pythia/pythia-410m-deduped.yml',
            data_prefix='/data/pile',tokenizer='/data/tokenizer',save_dir='/save',microbatch=8,
            checkpoint_stride=8,slim_checkpoints=True)
        self.assertEqual(1000,derived['exit-interval'])
        self.assertEqual(143000,derived['train-iters'])
        self.assertEqual(143000,derived['lr-decay-iters'])
        self.assertEqual(1024,derived['gas']*derived['train_micro_batch_size_per_gpu'])
        self.assertEqual(schedule,derived['extra-save-iters'])

    def test_vast_search_cannot_create_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            log = work / "args"
            fake = work / "vastai"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$VAST_LOG\"\n", encoding="utf-8")
            fake.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{work}:{env.get('PATH', '')}"
            env["VAST_LOG"] = str(log)
            run = subprocess.run(
                ["bash", str(REPRO / "vast_search.sh")],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, run.returncode, run.stderr)
            args = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(["search", "offers"], args[:2])
            self.assertNotIn("create", args)

    def test_checkpoint_retires_only_after_remote_hash_matches_and_receipt_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            checkpoint = work / "save" / "global_step8"
            checkpoint.mkdir(parents=True)
            (checkpoint / "layer_00-model_00-model_states.pt").write_bytes(b"weights")
            (checkpoint / ".gym-ready").write_text("8\n", encoding="utf-8")
            hub = FakeHub()

            receipt = self.retire.retire_checkpoint(
                checkpoint,
                receipts_dir=work / "receipts",
                staging_dir=work / "staging",
                hub=hub,
                remote_prefix="dense-0-512",
                retire=True,
            )
            self.assertFalse(checkpoint.exists())
            self.assertTrue(receipt["retired_local"])
            self.assertEqual(receipt["archive_sha256"], receipt["remote_lfs_sha256"])
            paths = [path for path, _ in hub.uploads]
            self.assertEqual(
                [
                    "dense-0-512/checkpoints/global_step8.tar",
                    "dense-0-512/receipts/step-000008.json",
                ],
                paths,
            )
            saved = json.loads((work / "receipts" / "step-000008.json").read_text())
            self.assertEqual(8, saved["step"])

    def test_hash_mismatch_never_retires_local_checkpoint(self) -> None:
        class BadHub(FakeHub):
            def remote_lfs_sha256(self, remote_path: str):
                return "0" * 64

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            checkpoint = work / "save" / "global_step16"
            checkpoint.mkdir(parents=True)
            (checkpoint / "layer.pt").write_bytes(b"weights")
            (checkpoint / ".gym-ready").write_text("16\n", encoding="utf-8")
            with self.assertRaises(self.retire.CheckpointError):
                self.retire.retire_checkpoint(
                    checkpoint,
                    receipts_dir=work / "receipts",
                    staging_dir=work / "staging",
                    hub=BadHub(),
                    remote_prefix="dense-0-512",
                )
            self.assertTrue(checkpoint.exists())

    def test_unready_checkpoint_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save = Path(directory)
            ready = save / "global_step32"
            ready.mkdir()
            (ready / "weights").write_bytes(b"x")
            (ready / ".gym-ready").write_text("32\n")
            partial = save / "global_step64"
            partial.mkdir()
            (partial / "weights").write_bytes(b"partial")
            self.assertEqual([ready], list(self.retire.ready_checkpoints(save)))

    def test_optimizer_state_blocks_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step128"
            checkpoint.mkdir()
            (checkpoint / ".gym-ready").write_text("128\n")
            (checkpoint / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"x")
            with self.assertRaises(self.retire.CheckpointError):
                self.retire.checkpoint_files(checkpoint)

    def test_verifier_help_does_not_import_torch(self) -> None:
        run = subprocess.run(
            ["python3", str(REPRO / "verify_anchor.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, run.returncode, run.stderr)
        self.assertIn("--checkpoint-dir", run.stdout)


if __name__ == "__main__":
    unittest.main()
