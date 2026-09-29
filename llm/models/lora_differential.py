from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ForwardRecord:
    loss: torch.Tensor
    logits: torch.Tensor
    activations: dict[str, torch.Tensor]
    adapter_input: torch.Tensor


class RankOneLoRA(nn.Module):
    """Minimal inspectable LoRA wrapper around one ``nn.Linear`` module."""

    def __init__(self, base: nn.Linear, *, rank: int = 1, alpha: float = 1.0, seed: int = 0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.merged = False
        self.last_input: torch.Tensor | None = None

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            torch.manual_seed(seed)
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return (self.b @ self.a) * self.scaling

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.a.detach().clone(),
            "b": self.b.detach().clone(),
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.a.copy_(state["a"].to(device=self.a.device, dtype=self.a.dtype))
            self.b.copy_(state["b"].to(device=self.b.device, dtype=self.b.dtype))

    def merge_(self) -> None:
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
        self.merged = True

    def unmerge_(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        base_output = self.base(x)
        if self.merged or not self.enabled:
            return base_output
        low_rank = F.linear(F.linear(x, self.a), self.b)
        return base_output + low_rank * self.scaling


def replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if not hasattr(parent, leaf):
        raise AttributeError(path)
    setattr(parent, leaf, replacement)


def attach_lora(
    model: nn.Module,
    target: str,
    *,
    rank: int = 1,
    alpha: float = 1.0,
    seed: int = 0,
) -> RankOneLoRA:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = model.get_submodule(target)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"{target} is {type(base).__name__}, expected nn.Linear")
    adapter = RankOneLoRA(base, rank=rank, alpha=alpha, seed=seed)
    replace_module(model, target, adapter)
    return adapter


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu()
    value_f = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": tensor_digest(value),
        "l2": float(torch.linalg.vector_norm(value_f).item()),
        "max_abs": float(value_f.abs().max().item()) if value.numel() else 0.0,
        "nonzero": int(torch.count_nonzero(value).item()),
    }


def frozen_parameter_hashes(model: nn.Module) -> dict[str, str]:
    return {
        name: tensor_digest(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError("module output contains no tensor")


def infer_transformer_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    for candidate in ("gpt_neox.layers", "model.layers", "transformer.h"):
        try:
            layers = model.get_submodule(candidate)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, nn.Sequential)):
            return [(f"{candidate}.{index}", layer) for index, layer in enumerate(layers)]
    return []


def forward_record(
    model: nn.Module,
    adapter: RankOneLoRA,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    require_grad: bool,
) -> ForwardRecord:
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            activations[name] = first_tensor(output).detach().clone()

        return hook

    for name, layer in infer_transformer_layers(model):
        hooks.append(layer.register_forward_hook(capture(name)))

    context = torch.enable_grad() if require_grad else torch.no_grad()
    try:
        with context:
            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss
            logits = output.logits
    finally:
        for hook in hooks:
            hook.remove()

    if adapter.last_input is None:
        raise RuntimeError("target adapter was not executed")
    return ForwardRecord(
        loss=loss,
        logits=logits.detach().clone(),
        activations=activations,
        adapter_input=adapter.last_input.detach().clone(),
    )


def save_tensor_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_tensor_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def run_test0(
    model: nn.Module,
    adapter: RankOneLoRA,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
) -> dict[str, Any]:
    frozen_before = frozen_parameter_hashes(model)

    attached = forward_record(model, adapter, input_ids, labels, require_grad=False)

    adapter.enabled = False
    disabled = forward_record(model, adapter, input_ids, labels, require_grad=False)
    adapter.enabled = True

    buffer = io.BytesIO()
    torch.save(adapter.adapter_state_dict(), buffer)
    buffer.seek(0)
    reloaded_state = torch.load(buffer, map_location=adapter.a.device, weights_only=True)
    adapter.load_adapter_state_dict(reloaded_state)
    reloaded = forward_record(model, adapter, input_ids, labels, require_grad=False)

    adapter.merge_()
    merged = forward_record(model, adapter, input_ids, labels, require_grad=False)
    adapter.unmerge_()
    unmerged = forward_record(model, adapter, input_ids, labels, require_grad=False)

    frozen_after = frozen_parameter_hashes(model)
    receipt = {
        "test": 0,
        "description": "adapter attached, no training",
        "checks": {
            "b_is_zero": bool(torch.count_nonzero(adapter.b).item() == 0),
            "attached_equals_disabled_logits": bool(torch.equal(attached.logits, disabled.logits)),
            "attached_equals_disabled_loss": bool(torch.equal(attached.loss.detach(), disabled.loss.detach())),
            "attached_equals_reloaded_logits": bool(torch.equal(attached.logits, reloaded.logits)),
            "attached_equals_merged_logits": bool(torch.equal(attached.logits, merged.logits)),
            "attached_equals_unmerged_logits": bool(torch.equal(attached.logits, unmerged.logits)),
            "attached_equals_disabled_activations": exact_tensor_dict_equal(attached.activations, disabled.activations),
            "frozen_parameters_unchanged": frozen_before == frozen_after,
        },
        "loss": float(attached.loss.detach().cpu().item()),
        "logits": tensor_summary(attached.logits),
        "adapter": {
            "a": tensor_summary(adapter.a),
            "b": tensor_summary(adapter.b),
            "delta_weight": tensor_summary(adapter.delta_weight()),
        },
    }
    receipt["passed"] = all(receipt["checks"].values())

    save_tensor_bundle(
        output_dir / "test-0.pt",
        {
            "attached_logits": attached.logits.cpu(),
            "disabled_logits": disabled.logits.cpu(),
            "reloaded_logits": reloaded.logits.cpu(),
            "merged_logits": merged.logits.cpu(),
            "unmerged_logits": unmerged.logits.cpu(),
            "attached_activations": {k: v.cpu() for k, v in attached.activations.items()},
            "disabled_activations": {k: v.cpu() for k, v in disabled.activations.items()},
            "a": adapter.a.detach().cpu(),
            "b": adapter.b.detach().cpu(),
            "delta_weight": adapter.delta_weight().detach().cpu(),
        },
    )
    write_json(output_dir / "test-0.json", receipt)
    return receipt


def run_optimizer_step(
    model: nn.Module,
    adapter: RankOneLoRA,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
) -> dict[str, Any]:
    frozen_before = frozen_parameter_hashes(model)
    a_before = adapter.a.detach().clone()
    b_before = adapter.b.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    before = forward_record(model, adapter, input_ids, labels, require_grad=True)
    before.loss.backward()
    if adapter.a.grad is None or adapter.b.grad is None:
        raise RuntimeError("adapter gradients were not produced")
    grad_a = adapter.a.grad.detach().clone()
    grad_b = adapter.b.grad.detach().clone()
    optimizer.step()

    a_after = adapter.a.detach().clone()
    b_after = adapter.b.detach().clone()
    after = forward_record(model, adapter, input_ids, labels, require_grad=False)
    frozen_after = frozen_parameter_hashes(model)

    delta_a = a_after - a_before
    delta_b = b_after - b_before
    delta_weight_before = (b_before @ a_before) * adapter.scaling
    delta_weight_after = adapter.delta_weight().detach().clone()
    delta_weight_step = delta_weight_after - delta_weight_before
    logit_delta = after.logits - before.logits
    activation_deltas = {
        name: after.activations[name] - before.activations[name]
        for name in before.activations.keys() & after.activations.keys()
    }

    x = after.adapter_input
    a_dot_x = F.linear(x, adapter.a.detach())
    injected = F.linear(a_dot_x, adapter.b.detach()) * adapter.scaling

    checks = {
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "adapter_changed": bool(torch.count_nonzero(delta_a).item() or torch.count_nonzero(delta_b).item()),
    }
    if step == 1 and torch.count_nonzero(b_before).item() == 0:
        checks["first_step_grad_a_zero"] = bool(torch.count_nonzero(grad_a).item() == 0)
        checks["first_step_grad_b_nonzero"] = bool(torch.count_nonzero(grad_b).item() != 0)

    receipt = {
        "test": 1 if step == 1 else 2,
        "optimizer_step": step,
        "checks": checks,
        "loss_before": float(before.loss.detach().cpu().item()),
        "loss_after": float(after.loss.detach().cpu().item()),
        "loss_delta": float((after.loss.detach() - before.loss.detach()).cpu().item()),
        "grad_a": tensor_summary(grad_a),
        "grad_b": tensor_summary(grad_b),
        "delta_a": tensor_summary(delta_a),
        "delta_b": tensor_summary(delta_b),
        "delta_weight_before": tensor_summary(delta_weight_before),
        "delta_weight_after": tensor_summary(delta_weight_after),
        "delta_weight_step": tensor_summary(delta_weight_step),
        "adapter_input": tensor_summary(x),
        "a_dot_x": tensor_summary(a_dot_x),
        "injected_direction": tensor_summary(injected),
        "logit_delta": tensor_summary(logit_delta),
        "activation_deltas": {name: tensor_summary(value) for name, value in activation_deltas.items()},
    }
    receipt["passed"] = all(checks.values())

    save_tensor_bundle(
        output_dir / f"step-{step:04d}.pt",
        {
            "a_before": a_before.cpu(),
            "b_before": b_before.cpu(),
            "grad_a": grad_a.cpu(),
            "grad_b": grad_b.cpu(),
            "delta_a": delta_a.cpu(),
            "delta_b": delta_b.cpu(),
            "delta_weight_before": delta_weight_before.cpu(),
            "delta_weight_after": delta_weight_after.cpu(),
            "delta_weight_step": delta_weight_step.cpu(),
            "adapter_input": x.cpu(),
            "a_dot_x": a_dot_x.cpu(),
            "injected_direction": injected.cpu(),
            "logit_delta": logit_delta.cpu(),
            "activation_deltas": {k: v.cpu() for k, v in activation_deltas.items()},
        },
    )
    write_json(output_dir / f"step-{step:04d}.json", receipt)
    return receipt


def run_steps(
    model: nn.Module,
    adapter: RankOneLoRA,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
    *,
    steps: int,
    learning_rate: float,
) -> list[dict[str, Any]]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    optimizer = torch.optim.SGD([adapter.a, adapter.b], lr=learning_rate)
    return [
        run_optimizer_step(model, adapter, input_ids, labels, optimizer, output_dir, step)
        for step in range(1, steps + 1)
    ]


def write_run_manifest(
    output_dir: Path,
    *,
    model_name: str,
    revision: str,
    target: str,
    rank: int,
    alpha: float,
    learning_rate: float,
    steps: int,
    seed: int,
    example: str,
    test0: dict[str, Any],
    step_receipts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    steps_payload = list(step_receipts)
    manifest = {
        "model": model_name,
        "revision": revision,
        "target": target,
        "rank": rank,
        "alpha": alpha,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "steps": steps,
        "seed": seed,
        "example": example,
        "test0": "test-0.json",
        "step_receipts": [f"step-{index:04d}.json" for index in range(1, len(steps_payload) + 1)],
        "passed": bool(test0["passed"] and all(item["passed"] for item in steps_payload)),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
