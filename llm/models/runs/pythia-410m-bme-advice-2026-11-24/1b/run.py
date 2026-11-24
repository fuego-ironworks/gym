"""Four-case 1B Pythia fallback using the checked 410M generation runner."""
from importlib import metadata
import json
from pathlib import Path
import sys

worktree = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(worktree / "llm/models"))
import run_pythia_bme_advice as baseline

baseline.MODEL = "EleutherAI/pythia-1b-deduped"
baseline.REVISION = "d1988145ec1cf1e76d786cd05af9f53e8b1c95cc"
baseline.MODEL_SHA256 = "3e51dc49d75de37ec6674b4a01cd0b45d058917c6eb82c610a68d09b05b90755"
baseline.TOKENIZER_BLOBS = {
    "config.json": "63fb5042229dbc3040adb966fd0ea24491bbd1b9",
    "special_tokens_map.json": "0967ef424bce6791893e9a57bb952f80fd536e93",
    "tokenizer.json": "08015a39463145a657e88a093aad79df9b2a76bf",
    "tokenizer_config.json": "c2df99e1a5c20585469000eb3c647617ab858232",
}


class LowMemoryLoader:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from torch import bfloat16
        from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
        kwargs["torch_dtype"] = bfloat16
        model = AutoModelForCausalLM.from_pretrained(*args, low_cpu_mem_usage=True, **kwargs)
        # The pinned tokenizer's end-of-text marker is ID 0. Model config ID 2 is '!'.
        if model.dtype != bfloat16:
            raise ValueError("expected bfloat16 model weights")
        model.generation_config.eos_token_id = 0
        class RequireFiniteScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if not scores.isfinite().all():
                    raise ValueError("nonfinite generation scores")
                return scores

        original_generate = model.generate

        def generate(*args, **options):
            return original_generate(
                *args, logits_processor=LogitsProcessorList([RequireFiniteScores()]), **options
            )

        model.generate = generate
        return model


class PythiaTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        if tokenizer.convert_tokens_to_ids("<|endoftext|>") != 0:
            raise ValueError("unexpected end-of-text token ID")
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        return tokenizer


baseline.AutoModelForCausalLM = LowMemoryLoader
baseline.AutoTokenizer = PythiaTokenizer
output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
baseline.main()
for name in ("plan.json", "receipt.json"):
    path = output_dir / name
    receipt = json.loads(path.read_text())
    receipt["precision"] = "bfloat16"
    receipt["versions"]["accelerate"] = metadata.version("accelerate")
    receipt["driver_sha256"] = baseline.file_sha256(Path(__file__))
    receipt["eos_token_id"] = 0
    receipt["finite_generation_scores"] = True
    baseline.atomic_json(path, receipt)
