#!/usr/bin/env python3
"""
Evaluation-only causal intervention on OLMoE router directions.

The model routes normally.  For tokens already grouped for expert ``i`` this
script optionally replaces the expert input ``x`` with

    x' = x - alpha * <x, d_i> d_i

immediately before the expert's gate/up projections.  ``d_i`` is either the
normalized router row or an expert-energy-matched direction orthogonal to that
row.  No model or checkpoint state is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from huggingface_hub import list_repo_refs, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.config import DType
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.moe.mlp import DroplessMoEMLP
from olmo_core.nn.moe.router import MoELinearRouter
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.ops import moe as moe_ops

SCHEMA_VERSION = 1
DEFAULT_REPO = "allenai/OLMoE-1B-7B-0924"
PILOT_STEPS = (5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 1_220_000)
LATE_GATE_STEPS = (250_000, 500_000, 1_000_000, 1_220_000)


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    label: str
    tokens: torch.Tensor


@dataclass(frozen=True)
class InterventionCondition:
    name: str
    direction_name: str
    alpha: float
    restore_norm: bool = False


@dataclass
class ControlBundle:
    router: torch.Tensor
    candidates: torch.Tensor
    candidate_eigenvalues: torch.Tensor
    router_quadratic_energy: torch.Tensor
    matched: torch.Tensor | None = None
    hard: torch.Tensor | None = None
    weighted_0: torch.Tensor | None = None
    weighted_1: torch.Tensor | None = None
    weighted_2: torch.Tensor | None = None
    calibration: dict[str, Any] | None = None

    def direction(self, name: str) -> torch.Tensor:
        value = getattr(self, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"direction '{name}' has not been selected")
        return value


@dataclass
class RouteCapture:
    indices_hash: str
    counts_hash: str
    weights: torch.Tensor

    def serializable(self) -> dict[str, Any]:
        rounded = torch.round(self.weights * 1_000_000) / 1_000_000
        return {
            "indices_hash": self.indices_hash,
            "counts_hash": self.counts_hash,
            "weights_1e-6_hash": hashlib.sha256(rounded.numpy().tobytes()).hexdigest(),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "weight_mean": float(self.weights.mean()),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_inference_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    _bin_ids: torch.Tensor,
    _bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Pure PyTorch equivalent of the CUDA dropless gather for inference."""
    token_indices = indices.to(torch.long) // top_k
    return x.index_select(0, token_indices)


def torch_inference_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    _bin_ids: torch.Tensor,
    weights: torch.Tensor | None,
    _bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Pure PyTorch equivalent of the CUDA dropless scatter for inference."""
    if weights is None:
        raise ValueError("dropless inference scatter requires expert weights")
    assignment_indices = indices.to(torch.long)
    token_indices = assignment_indices // top_k
    num_tokens = weights.numel() // top_k
    out = torch.zeros(num_tokens, x.shape[-1], dtype=x.dtype, device=x.device)
    selected_weights = weights.index_select(0, assignment_indices).to(x.dtype).unsqueeze(-1)
    out.index_add_(0, token_indices, x * selected_weights)
    return out


def install_noncuda_moe_inference_fallback(device: torch.device) -> bool:
    """Install process-local routing permutations only when CUDA kernels cannot be used."""
    if device.type == "cuda":
        return False
    moe_ops.gather = torch_inference_gather
    moe_ops.scatter = torch_inference_scatter
    return True


def resolve_revision(repo: str, revision: str) -> str:
    if not revision.startswith("step:"):
        return revision
    target = int(revision.split(":", 1)[1])
    prefix = f"step{target}-"
    matches = sorted(
        ref.name for ref in list_repo_refs(repo).branches if ref.name.startswith(prefix)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one public branch starting with '{prefix}', found {matches}"
        )
    return matches[0]


def public_step_revisions(repo: str) -> list[str]:
    refs = []
    for ref in list_repo_refs(repo).branches:
        match = re.fullmatch(r"step(\d+)-tokens.+", ref.name)
        if match:
            refs.append((int(match.group(1)), ref.name))
    return [name for _, name in sorted(refs)]


def parse_step(revision: str) -> int:
    match = re.search(r"(?:^|/)step(\d+)(?:-|$)", revision)
    if match is None:
        raise ValueError(f"cannot parse optimizer step from revision '{revision}'")
    return int(match.group(1))


def normalized_router_directions(router: MoELinearRouter) -> torch.Tensor:
    rows = router.weight.detach().view(router.num_experts, router.d_model).float()
    norms = rows.norm(dim=-1, keepdim=True)
    if torch.any(norms <= torch.finfo(rows.dtype).tiny):
        raise RuntimeError("router contains a zero-norm row")
    return rows / norms


def expert_weight_views(mlp: DroplessMoEMLP) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        mlp.w1.detach().view(mlp.num_experts, mlp.hidden_size, mlp.d_model),
        mlp.w2.detach().view(mlp.num_experts, mlp.hidden_size, mlp.d_model),
        mlp.w3.detach().view(mlp.num_experts, mlp.hidden_size, mlp.d_model),
    )


def apply_grouped_projection(
    x: torch.Tensor,
    batch_size_per_expert: torch.Tensor,
    directions: torch.Tensor,
    *,
    alpha: float,
    restore_norm: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one direction per contiguous expert group."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 0.0:
        return x, {
            "projection_residual_max": 0.0,
            "removed_component_rms": 0.0,
            "norm_ratio_mean": 1.0,
        }
    if directions.shape != (batch_size_per_expert.numel(), x.shape[-1]):
        raise ValueError(
            f"directions shape {tuple(directions.shape)} does not match "
            f"({batch_size_per_expert.numel()}, {x.shape[-1]})"
        )

    counts = batch_size_per_expert.to(device=x.device, dtype=torch.long)
    expert_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=x.device), counts, output_size=x.shape[0]
    )
    dirs = directions.to(device=x.device, dtype=torch.float32)[expert_ids]
    x_float = x.float()
    component = (x_float * dirs).sum(dim=-1, keepdim=True)
    projected = x_float - alpha * component * dirs

    if restore_norm:
        old_norm = x_float.norm(dim=-1, keepdim=True)
        new_norm = projected.norm(dim=-1, keepdim=True)
        scale = torch.where(
            new_norm > torch.finfo(projected.dtype).eps,
            old_norm / new_norm.clamp_min(torch.finfo(projected.dtype).eps),
            torch.ones_like(new_norm),
        )
        projected = projected * scale

    expected_component = (1.0 - alpha) * component
    actual_component = (projected * dirs).sum(dim=-1, keepdim=True)
    residual = (actual_component - expected_component).abs().max()
    removed_rms = (alpha * component).square().mean().sqrt()
    norm_ratio = projected.norm(dim=-1) / x_float.norm(dim=-1).clamp_min(
        torch.finfo(projected.dtype).eps
    )
    diagnostics = {
        "projection_residual_max": float(residual.detach().cpu()),
        "removed_component_rms": float(removed_rms.detach().cpu()),
        "norm_ratio_mean": float(norm_ratio.mean().detach().cpu()),
    }
    return projected.to(dtype=x.dtype), diagnostics


def _project_away_router(vectors: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    return vectors - torch.einsum("ekd,ed->ek", vectors, router).unsqueeze(-1) * router.unsqueeze(1)


def _batched_orthonormalize(vectors: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
    vectors = _project_away_router(vectors, router)
    # Modified Gram-Schmidt avoids ``torch.linalg.qr``, which is unavailable on
    # MPS and would force the real local smoke onto a different implementation.
    basis: list[torch.Tensor] = []
    tiny = torch.finfo(vectors.dtype).eps
    for idx in range(vectors.shape[1]):
        vector = vectors[:, idx]
        for previous in basis:
            vector = vector - (vector * previous).sum(dim=-1, keepdim=True) * previous
        norm = vector.norm(dim=-1, keepdim=True)
        if torch.any(norm <= tiny):
            raise RuntimeError("block power iteration produced a degenerate control direction")
        basis.append(vector / norm)
    return torch.stack(basis, dim=1)


def apply_expert_energy_operator(
    vectors: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    """Apply ``W1.T W1 + W3.T W3`` without constructing the square matrix."""
    w1f = w1.float()
    w3f = w3.float()
    hidden1 = torch.bmm(vectors, w1f.transpose(1, 2))
    hidden3 = torch.bmm(vectors, w3f.transpose(1, 2))
    return torch.bmm(hidden1, w1f) + torch.bmm(hidden3, w3f)


def quadratic_energy(directions: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    single = directions.unsqueeze(1)
    return (single * apply_expert_energy_operator(single, w1, w3)).sum(dim=-1).squeeze(1)


@torch.no_grad()
def implicit_orthogonal_eigendirections(
    w1: torch.Tensor,
    w3: torch.Tensor,
    router: torch.Tensor,
    *,
    candidate_count: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block power iteration for high-energy directions in ``router.T``'s orthogonal complement."""
    num_experts, _, d_model = w1.shape
    if not 1 <= candidate_count < d_model:
        raise ValueError("candidate_count must be in [1, d_model)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors = torch.randn(
        num_experts,
        candidate_count,
        d_model,
        generator=generator,
        dtype=torch.float32,
    ).to(router.device)
    vectors = _batched_orthonormalize(vectors, router)
    for _ in range(iterations):
        vectors = apply_expert_energy_operator(vectors, w1, w3)
        vectors = _batched_orthonormalize(vectors, router)
    applied = apply_expert_energy_operator(vectors, w1, w3)
    eigenvalues = (vectors * applied).sum(dim=-1).clamp_min(0)
    order = eigenvalues.argsort(dim=-1, descending=True)
    vectors = vectors.gather(1, order.unsqueeze(-1).expand(-1, -1, vectors.shape[-1]))
    eigenvalues = eigenvalues.gather(1, order)
    orthogonality = torch.einsum("ekd,ed->ek", vectors, router).abs().max()
    if float(orthogonality.cpu()) > 2e-4:
        raise RuntimeError(f"control directions are not orthogonal enough: {orthogonality}")
    return vectors, eigenvalues


class ActivationMomentHook:
    def __init__(self, mlp: DroplessMoEMLP, bundle: ControlBundle):
        self.mlp = mlp
        self.bundle = bundle
        device = bundle.router.device
        self.router_sum_sq = torch.zeros(mlp.num_experts, device=device)
        self.candidate_sum_sq = torch.zeros(
            mlp.num_experts, bundle.candidates.shape[1], device=device
        )
        self.count = torch.zeros(mlp.num_experts, device=device)
        self.handle: Any = None

    def __enter__(self) -> Self:
        self.handle = self.mlp.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    @torch.no_grad()
    def _hook(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        x, counts = args
        counts = counts.to(device=x.device, dtype=torch.long)
        expert_ids = torch.repeat_interleave(
            torch.arange(counts.numel(), device=x.device), counts, output_size=x.shape[0]
        )
        x_float = x.float()
        router = self.bundle.router.to(x.device)[expert_ids]
        candidates = self.bundle.candidates.to(x.device)[expert_ids]
        router_dot = (x_float * router).sum(dim=-1)
        candidate_dot = torch.einsum("nd,nkd->nk", x_float, candidates)
        self.router_sum_sq.index_add_(0, expert_ids, router_dot.square())
        self.candidate_sum_sq.index_add_(0, expert_ids, candidate_dot.square())
        self.count.add_(counts)

    def moments(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = self.count.clamp_min(1)
        return (
            self.router_sum_sq / denom,
            self.candidate_sum_sq / denom.unsqueeze(-1),
            self.count,
        )


def select_calibrated_controls(
    bundle: ControlBundle,
    moment_hook: ActivationMomentHook,
    *,
    weighted_seeds: Sequence[int] = (1701, 1702, 1703),
) -> None:
    router_moment, candidate_moment, counts = moment_hook.moments()
    router_eta = bundle.router_quadratic_energy * router_moment
    candidate_eta = bundle.candidate_eigenvalues * candidate_moment
    matched_idx = (candidate_eta - router_eta.unsqueeze(-1)).abs().argmin(dim=-1)
    hard_idx = candidate_eta.argmax(dim=-1)
    expert_idx = torch.arange(bundle.router.shape[0], device=bundle.router.device)
    bundle.matched = bundle.candidates[expert_idx, matched_idx]
    bundle.hard = bundle.candidates[expert_idx, hard_idx]

    weighted: list[torch.Tensor] = []
    probabilities = candidate_eta.clamp_min(0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).eps
    )
    uniform_rows = candidate_eta.sum(dim=-1) <= 0
    if uniform_rows.any():
        probabilities[uniform_rows] = 1.0 / probabilities.shape[-1]
    for seed in weighted_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        sampled = (
            torch.multinomial(
                probabilities.detach().cpu(), 1, replacement=True, generator=generator
            )
            .squeeze(-1)
            .to(bundle.router.device)
        )
        weighted.append(bundle.candidates[expert_idx, sampled])
    bundle.weighted_0, bundle.weighted_1, bundle.weighted_2 = weighted
    bundle.calibration = {
        "counts": counts,
        "router_second_moment": router_moment,
        "candidate_second_moment": candidate_moment,
        "router_eta": router_eta,
        "candidate_eta": candidate_eta,
        "matched_index": matched_idx,
        "hard_index": hard_idx,
        "matched_eta_ratio": candidate_eta[expert_idx, matched_idx]
        / router_eta.clamp_min(torch.finfo(router_eta.dtype).eps),
    }


class RouteFingerprintHook:
    def __init__(self, router: MoELinearRouter):
        self.router = router
        self.handle: Any = None
        self._indices: list[torch.Tensor] | None = None
        self._weights: list[torch.Tensor] | None = None
        self._counts: list[torch.Tensor] | None = None

    def __enter__(self) -> Self:
        self.handle = self.router.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def begin(self) -> None:
        self._indices = []
        self._weights = []
        self._counts = []

    def end(self) -> RouteCapture:
        if self._indices is None or self._weights is None or self._counts is None:
            raise RuntimeError("route fingerprint was not started")
        indices = torch.cat(self._indices)
        weights = torch.cat(self._weights)
        counts = torch.cat(self._counts)
        capture = RouteCapture(
            indices_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            counts_hash=hashlib.sha256(counts.numpy().tobytes()).hexdigest(),
            weights=weights,
        )
        self._indices = None
        self._weights = None
        self._counts = None
        return capture

    @torch.no_grad()
    def _hook(
        self, _module: torch.nn.Module, _args: tuple[Any, ...], output: tuple[Any, ...]
    ) -> None:
        if self._indices is None or self._weights is None or self._counts is None:
            return
        weights, indices, counts, _aux = output
        self._indices.append(indices.detach().to(torch.int32).contiguous().cpu().reshape(-1))
        self._weights.append(weights.detach().float().contiguous().cpu().reshape(-1))
        self._counts.append(counts.detach().to(torch.int64).contiguous().cpu().reshape(-1))


def serialize_route_captures(
    captures: Mapping[str, Sequence[RouteCapture]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        layer: [capture.serializable() for capture in layer_captures]
        for layer, layer_captures in captures.items()
    }


def compare_route_captures(
    left: Mapping[str, Sequence[RouteCapture]],
    right: Mapping[str, Sequence[RouteCapture]],
) -> dict[str, Any]:
    if left.keys() != right.keys():
        return {
            "ids_and_counts_identical": False,
            "weights_max_abs_error": math.inf,
        }
    exact = True
    max_weight_error = 0.0
    for layer in left:
        if len(left[layer]) != len(right[layer]):
            exact = False
            continue
        for first, second in zip(left[layer], right[layer]):
            exact &= first.indices_hash == second.indices_hash
            exact &= first.counts_hash == second.counts_hash
            if first.weights.shape != second.weights.shape:
                max_weight_error = math.inf
            else:
                max_weight_error = max(
                    max_weight_error, float((first.weights - second.weights).abs().max())
                )
    return {
        "ids_and_counts_identical": exact,
        "weights_max_abs_error": max_weight_error,
    }


class GroupedProjectionHook:
    def __init__(
        self,
        mlp: DroplessMoEMLP,
        directions: torch.Tensor,
        condition: InterventionCondition,
        *,
        validate_manual: bool = False,
    ):
        self.mlp = mlp
        self.directions = directions
        self.condition = condition
        self.validate_manual = validate_manual
        self.pre_handle: Any = None
        self.post_handle: Any = None
        self.projection_residual_max = 0.0
        self.removed_component_sum = 0.0
        self.norm_ratio_sum = 0.0
        self.calls = 0
        self.manual_output_max_abs_error = 0.0
        self._manual_done = False

    def __enter__(self) -> Self:
        self.pre_handle = self.mlp.register_forward_pre_hook(self._pre_hook)
        if self.validate_manual:
            self.post_handle = self.mlp.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.pre_handle is not None:
            self.pre_handle.remove()
            self.pre_handle = None
        if self.post_handle is not None:
            self.post_handle.remove()
            self.post_handle = None

    def _pre_hook(
        self, _module: torch.nn.Module, args: tuple[Any, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, counts = args
        projected, diagnostics = apply_grouped_projection(
            x,
            counts,
            self.directions,
            alpha=self.condition.alpha,
            restore_norm=self.condition.restore_norm,
        )
        self.projection_residual_max = max(
            self.projection_residual_max, diagnostics["projection_residual_max"]
        )
        self.removed_component_sum += diagnostics["removed_component_rms"]
        self.norm_ratio_sum += diagnostics["norm_ratio_mean"]
        self.calls += 1
        return projected, counts

    @torch.no_grad()
    def _post_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        if self._manual_done or output.numel() == 0:
            return
        assert isinstance(module, DroplessMoEMLP)
        x, counts = args
        nonempty = torch.nonzero(counts > 0, as_tuple=False)
        if nonempty.numel() == 0:
            return
        expert = int(nonempty[0].item())
        start = int(counts[:expert].sum().item())
        token = x[start : start + 1]
        w1, w2, w3 = expert_weight_views(module)
        token_compute = token.to(w1.dtype)
        hidden = F.silu(token_compute @ w1[expert].t()) * (token_compute @ w3[expert].t())
        expected = hidden @ w2[expert]
        actual = output[start : start + 1]
        self.manual_output_max_abs_error = float(
            (expected.to(actual.dtype) - actual).abs().max().cpu()
        )
        self._manual_done = True

    def diagnostics(self) -> dict[str, float]:
        denom = max(self.calls, 1)
        return {
            "projection_residual_max": self.projection_residual_max,
            "removed_component_rms_mean": self.removed_component_sum / denom,
            "norm_ratio_mean": self.norm_ratio_sum / denom,
            "manual_output_max_abs_error": self.manual_output_max_abs_error,
            "hook_calls": float(self.calls),
        }


def condition_set(name: str) -> list[InterventionCondition]:
    identity = InterventionCondition("identity_alpha0", "router", 0.0)
    router_05 = InterventionCondition("router_alpha0.5", "router", 0.5)
    router_10 = InterventionCondition("router_alpha1", "router", 1.0)
    matched_05 = InterventionCondition("matched_alpha0.5", "matched", 0.5)
    matched_10 = InterventionCondition("matched_alpha1", "matched", 1.0)
    if name == "full":
        return [router_10, matched_10]
    smoke = [identity, router_05, router_10, matched_10]
    if name == "smoke":
        return smoke
    if name in {"pilot", "final"}:
        return [
            identity,
            router_05,
            router_10,
            matched_05,
            matched_10,
            InterventionCondition("router_alpha1_norm", "router", 1.0, True),
            InterventionCondition("matched_alpha1_norm", "matched", 1.0, True),
            InterventionCondition("hard_alpha1", "hard", 1.0),
            InterventionCondition("weighted0_alpha1", "weighted_0", 1.0),
            InterventionCondition("weighted1_alpha1", "weighted_1", 1.0),
            InterventionCondition("weighted2_alpha1", "weighted_2", 1.0),
        ]
    raise ValueError(f"unknown condition set '{name}'")


def _repeat_to_length(tokens: Sequence[int], length: int) -> torch.Tensor:
    if not tokens:
        raise ValueError("tokenizer returned no tokens")
    repeats = math.ceil(length / len(tokens))
    return torch.tensor((list(tokens) * repeats)[:length], dtype=torch.long)


def local_smoke_samples(
    tokenizer: Any, sequence_length: int
) -> tuple[list[EvalSample], list[EvalSample]]:
    calibration_texts = [
        "Mixture of experts models route each token to a small subset of specialized networks.",
        "A causal intervention should preserve the routing decision while changing only expert computation.",
    ]
    evaluation_texts = [
        "Jerusalem has a long history, and its streets contain architecture from many periods.",
        "Careful experiments compare an intervention with a matched control instead of relying on correlation.",
    ]
    calibration = [
        EvalSample(
            f"local-cal-{idx}",
            "local-calibration",
            _repeat_to_length(
                tokenizer.encode(text, add_special_tokens=False), sequence_length + 1
            ),
        )
        for idx, text in enumerate(calibration_texts)
    ]
    evaluation = [
        EvalSample(
            f"local-eval-{idx}",
            "local",
            _repeat_to_length(
                tokenizer.encode(text, add_special_tokens=False), sequence_length + 1
            ),
        )
        for idx, text in enumerate(evaluation_texts)
    ]
    return calibration, evaluation


def parse_dataset_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset spec must be LABEL=PATH, got '{value}'")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not label or not path.is_file():
            raise ValueError(f"invalid dataset spec '{value}'")
        specs.append((label, path))
    if not specs:
        raise ValueError("at least one --dataset-spec is required outside --local-smoke")
    return specs


def numpy_validation_samples(
    specs: Sequence[tuple[str, Path]],
    *,
    sequence_length: int,
    calibration_per_label: int,
    evaluation_per_label: int,
) -> tuple[list[EvalSample], list[EvalSample]]:
    by_label: dict[str, list[Path]] = {}
    for label, path in specs:
        by_label.setdefault(label, []).append(path)
    calibration: list[EvalSample] = []
    evaluation: list[EvalSample] = []
    width = sequence_length + 1
    needed = calibration_per_label + evaluation_per_label
    for label, paths in sorted(by_label.items()):
        chunks: list[np.ndarray] = []
        for path in paths:
            array = np.load(path, mmap_mode="r")
            flat = np.asarray(array).reshape(-1)
            usable = (flat.size // width) * width
            if usable:
                chunks.extend(np.split(np.asarray(flat[:usable]), usable // width))
            if len(chunks) >= needed:
                break
        if len(chunks) < needed:
            raise RuntimeError(
                f"label '{label}' has only {len(chunks)} sequences of length {width}, needs {needed}"
            )
        for idx, chunk in enumerate(chunks[:calibration_per_label]):
            calibration.append(
                EvalSample(
                    f"{label}-cal-{idx:04d}",
                    label,
                    torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True)),
                )
            )
        start = calibration_per_label
        for idx, chunk in enumerate(chunks[start : start + evaluation_per_label]):
            evaluation.append(
                EvalSample(
                    f"{label}-eval-{idx:04d}",
                    label,
                    torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True)),
                )
            )
    return calibration, evaluation


def target_modules(
    model: torch.nn.Module, layers: Sequence[int]
) -> dict[int, tuple[MoELinearRouter, DroplessMoEMLP]]:
    selected: dict[int, tuple[MoELinearRouter, DroplessMoEMLP]] = {}
    blocks = model.blocks
    for layer in layers:
        block = blocks[str(layer)]
        moe = block.feed_forward_moe
        router = moe.router
        mlp = moe.experts.mlp
        if not isinstance(router, MoELinearRouter):
            raise TypeError(
                f"layer {layer} router is {type(router).__name__}, expected MoELinearRouter"
            )
        if not isinstance(mlp, DroplessMoEMLP):
            raise TypeError(
                f"layer {layer} experts are {type(mlp).__name__}, expected DroplessMoEMLP"
            )
        selected[layer] = (router, mlp)
    return selected


def build_control_bundle(
    router: MoELinearRouter,
    mlp: DroplessMoEMLP,
    *,
    candidate_count: int,
    power_iterations: int,
    seed: int,
) -> ControlBundle:
    router_directions = normalized_router_directions(router)
    w1, _w2, w3 = expert_weight_views(mlp)
    candidates, eigenvalues = implicit_orthogonal_eigendirections(
        w1,
        w3,
        router_directions,
        candidate_count=candidate_count,
        iterations=power_iterations,
        seed=seed,
    )
    router_energy = quadratic_energy(router_directions, w1, w3)
    return ControlBundle(
        router=router_directions,
        candidates=candidates,
        candidate_eigenvalues=eigenvalues,
        router_quadratic_energy=router_energy,
    )


@torch.no_grad()
def run_calibration(
    model: torch.nn.Module,
    samples: Sequence[EvalSample],
    hooks: Mapping[int, ActivationMomentHook],
    *,
    device: torch.device,
) -> None:
    with contextlib.ExitStack() as stack:
        for hook in hooks.values():
            stack.enter_context(hook)
        for sample in samples:
            tokens = sample.tokens.to(device)
            input_ids = tokens[:-1].unsqueeze(0)
            labels = tokens[1:].unsqueeze(0)
            model(
                input_ids=input_ids,
                labels=labels,
                loss_reduction="none",
                return_logits=False,
            )


@torch.no_grad()
def evaluate_samples(
    model: torch.nn.Module,
    samples: Sequence[EvalSample],
    *,
    device: torch.device,
    route_hooks: Mapping[int, RouteFingerprintHook],
    keep_logits: bool,
) -> tuple[dict[str, Any], dict[str, list[RouteCapture]], list[torch.Tensor] | None]:
    labels = sorted({sample.label for sample in samples})
    # MeanMetric's scalar in-place accumulators do not update on MPS. Keeping
    # metrics on CPU is backend-independent; model execution remains on `device`.
    evaluator = LMEvaluator(
        name="router-axis", batches=(), labels=labels, device=torch.device("cpu")
    )
    per_sequence: dict[str, float] = {}
    direct_by_label: dict[str, list[float]] = {label: [] for label in labels}
    route_fingerprints: dict[str, list[RouteCapture]] = {str(layer): [] for layer in route_hooks}
    logits_out: list[torch.Tensor] | None = [] if keep_logits else None
    direct_ce_max_abs_error = 0.0

    with contextlib.ExitStack() as stack:
        active_routes = {layer: stack.enter_context(hook) for layer, hook in route_hooks.items()}
        for sample in samples:
            for hook in active_routes.values():
                hook.begin()
            tokens = sample.tokens.to(device)
            input_ids = tokens[:-1].unsqueeze(0)
            labels_tensor = tokens[1:].unsqueeze(0)
            output = model(
                input_ids=input_ids,
                labels=labels_tensor,
                loss_reduction="none",
                return_logits=keep_logits,
            )
            ce_loss = output.ce_loss.view_as(labels_tensor)
            batch = {
                "metadata": [{"label": sample.label, "sample_id": sample.sample_id}],
                "label_mask": torch.ones_like(labels_tensor, dtype=torch.bool),
            }
            evaluator.update_metrics(batch, ce_loss, output.logits)
            value = float(ce_loss.mean().cpu())
            per_sequence[sample.sample_id] = value
            direct_by_label[sample.label].append(value)
            for layer, hook in active_routes.items():
                route_fingerprints[str(layer)].append(hook.end())
            if logits_out is not None:
                assert output.logits is not None
                logits = output.logits.detach().cpu()
                logits_out.append(logits)
                direct = F.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]),
                    labels_tensor.cpu().reshape(-1),
                    reduction="none",
                ).view_as(ce_loss.cpu())
                direct_ce_max_abs_error = max(
                    direct_ce_max_abs_error, float((direct - ce_loss.cpu()).abs().max())
                )

    native_metrics = {key: float(value.cpu()) for key, value in evaluator.compute_metrics().items()}
    evaluator_direct_max_abs_error = 0.0
    for label, values in direct_by_label.items():
        evaluator_direct_max_abs_error = max(
            evaluator_direct_max_abs_error,
            abs(native_metrics[f"{label}/CE loss"] - float(np.mean(values))),
        )
    record = {
        "per_sequence_ce": per_sequence,
        "metrics": native_metrics,
        "mean_ce": float(np.mean(list(per_sequence.values()))),
        "mean_ppl": float(math.exp(np.mean(list(per_sequence.values())))),
        "direct_ce_max_abs_error": direct_ce_max_abs_error,
        "evaluator_direct_max_abs_error": evaluator_direct_max_abs_error,
    }
    return record, route_fingerprints, logits_out


def _paired_values(first: Mapping[str, float], second: Mapping[str, float]) -> np.ndarray:
    keys = sorted(set(first) & set(second))
    if not keys:
        raise RuntimeError("paired records have no common samples")
    return np.array([first[key] - second[key] for key in keys], dtype=np.float64)


def paired_bootstrap_ci(
    values: np.ndarray, *, seed: int = 19_871, draws: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def add_layer_statistics(layer_record: dict[str, Any], baseline: Mapping[str, Any]) -> None:
    baseline_ce = baseline["per_sequence_ce"]
    for condition in layer_record["conditions"].values():
        delta = _paired_values(condition["per_sequence_ce"], baseline_ce)
        condition["delta_ce_mean"] = float(delta.mean())
        condition["delta_ce_ci95"] = paired_bootstrap_ci(delta)
    conditions = layer_record["conditions"]
    for suffix in ("0.5", "1"):
        router_key = f"router_alpha{suffix}"
        matched_key = f"matched_alpha{suffix}"
        if router_key in conditions and matched_key in conditions:
            d_values = _paired_values(
                conditions[router_key]["per_sequence_ce"],
                conditions[matched_key]["per_sequence_ce"],
            )
            layer_record[f"D_alpha{suffix}"] = {
                "mean": float(d_values.mean()),
                "ci95": paired_bootstrap_ci(d_values),
                "per_sequence": d_values.tolist(),
            }


def load_native_model(
    *,
    repo: str,
    revision: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[torch.nn.Module, Any, Path]:
    snapshot = Path(snapshot_download(repo, revision=revision, local_files_only=local_files_only))
    hf_config = AutoConfig.from_pretrained(snapshot)
    config = TransformerConfig.olmoe_1B_7B(
        vocab_size=hf_config.vocab_size,
        dtype=DType.bfloat16,
    )
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.feed_forward_moe is not None
    config.block.feed_forward_moe.dtype = DType.bfloat16
    model = config.build(init_device="cpu")
    options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True,
        cpu_offload=True,
    )
    state = dist_cp_sd.get_model_state_dict(model, options=options)
    hf_model = AutoModelForCausalLM.from_pretrained(snapshot)
    hf_model.resize_token_embeddings(hf_config.vocab_size)
    hf_state = dict(hf_model.state_dict())
    fused_experts = extract_fused_olmoe_expert_state(hf_state, model.state_dict())
    feed_forward_norms = extract_olmoe_feed_forward_norm_state(hf_state, model.state_dict())
    converted_state = convert_state_from_hf(
        hf_model.config,
        hf_state,
        model_type=getattr(hf_model.config, "model_type", None),
    )
    converted_state.update(fused_experts)
    converted_state.update(feed_forward_norms)
    repair_olmoe_gate_up_layout(converted_state, model.state_dict())
    missing = sorted(set(state) - set(converted_state))
    unexpected = sorted(set(converted_state) - set(state))
    if missing or unexpected:
        raise RuntimeError(f"native/HF state mismatch: missing={missing}, unexpected={unexpected}")
    state.update(converted_state)
    model.load_state_dict(state, assign=True)
    del converted_state, feed_forward_norms, fused_experts, hf_state, hf_model, state
    gc.collect()
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    return model, tokenizer, snapshot


def extract_olmoe_feed_forward_norm_state(
    hf_state: dict[str, Any], native_state: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """
    Map OLMoE's post-attention norm to the native pre-MoE norm.

    The generic upstream HF mapping aliases both ``input_layernorm`` and
    ``post_attention_layernorm`` to ``attention_norm`` and otherwise leaves
    native ``feed_forward_norm`` at random initialization.  Remove the latter
    source key before generic conversion and map it explicitly here.
    """
    converted: dict[str, torch.Tensor] = {}
    pattern = re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$")
    for hf_key in sorted(hf_state):
        match = pattern.match(hf_key)
        if match is None:
            continue
        layer = int(match.group(1))
        value = hf_state.pop(hf_key)
        native_key = f"blocks.{layer}.feed_forward_norm.weight"
        target = native_state[native_key]
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or value.shape != target.shape
        ):
            raise RuntimeError(
                f"cannot map {hf_key} to {native_key}: "
                f"value={getattr(value, 'shape', None)}, native={getattr(target, 'shape', None)}"
            )
        converted[native_key] = value
    return converted


def extract_fused_olmoe_expert_state(
    hf_state: dict[str, Any], native_state: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """
    Convert the fused expert layout used by newer Transformers OLMoE modules.

    Some public revisions load as ``gate_up_proj[E, 2H, D]`` and
    ``down_proj[E, D, H]`` instead of per-expert Linear modules.  Remove only
    those fused keys from the temporary HF state and return their exact native
    OLMo-core W1/W2/W3 tensors.  The checkpoint and converter are untouched.
    """
    converted: dict[str, torch.Tensor] = {}
    pattern = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
    for gate_key in sorted(hf_state):
        match = pattern.match(gate_key)
        if match is None:
            continue
        layer = int(match.group(1))
        down_key = f"model.layers.{layer}.mlp.experts.down_proj"
        if down_key not in hf_state:
            raise RuntimeError(f"missing fused expert tensor {down_key}")
        gate_up = hf_state.pop(gate_key)
        down = hf_state.pop(down_key)
        if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
            raise TypeError(f"fused expert states for layer {layer} must be tensors")
        if gate_up.ndim != 3 or down.ndim != 3:
            raise RuntimeError(
                f"unexpected fused expert ranks at layer {layer}: "
                f"gate/up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
            )
        num_experts, twice_hidden, d_model = gate_up.shape
        if twice_hidden % 2:
            raise RuntimeError(f"odd fused gate/up hidden dimension at layer {layer}")
        hidden_size = twice_hidden // 2
        if down.shape != (num_experts, d_model, hidden_size):
            raise RuntimeError(
                f"incompatible fused down projection at layer {layer}: "
                f"gate/up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
            )
        gate, up = gate_up.split(hidden_size, dim=1)
        prefix = f"blocks.{layer}.feed_forward_moe.experts.mlp"
        native_values = {
            f"{prefix}.w1": gate.contiguous(),
            f"{prefix}.w2": down.transpose(1, 2).contiguous(),
            f"{prefix}.w3": up.contiguous(),
        }
        for key, value in native_values.items():
            target = native_state[key]
            if not isinstance(target, torch.Tensor) or value.numel() != target.numel():
                raise RuntimeError(
                    f"cannot map fused {key}: value={tuple(value.shape)}, "
                    f"native={getattr(target, 'shape', None)}"
                )
            converted[key] = value.view_as(target)
    dangling = sorted(
        key for key in hf_state if re.match(r"^model\.layers\.\d+\.mlp\.experts\.down_proj$", key)
    )
    if dangling:
        raise RuntimeError(f"fused down projections without gate/up tensors: {dangling}")
    return converted


def repair_olmoe_gate_up_layout(
    converted_state: dict[str, Any], native_state: Mapping[str, Any]
) -> list[str]:
    """
    Adapt the HF converter's expert-major gate/up layout without touching the converter.

    For OLMoE, converted W1/W3 tensors currently arrive as ``(E*D, H)`` while
    :class:`DroplessMoEMLP` stores them as ``(E*H, D)``.  The values are already
    ordered by expert, so this is a per-expert transpose.  W2 already has the
    native layout and is deliberately untouched.
    """
    repaired: list[str] = []
    for key, source in list(converted_state.items()):
        if not re.search(r"\.experts\.mlp\.w[13]$", key):
            continue
        target = native_state[key]
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if source.shape == target.shape:
            continue
        if source.ndim != 2 or target.ndim != 2 or source.numel() != target.numel():
            raise RuntimeError(
                f"cannot repair {key}: converted {tuple(source.shape)}, native {tuple(target.shape)}"
            )
        hidden_size = source.shape[1]
        d_model = target.shape[1]
        if target.shape[0] % hidden_size != 0:
            raise RuntimeError(f"cannot infer expert count for {key}")
        num_experts = target.shape[0] // hidden_size
        if source.shape[0] != num_experts * d_model:
            raise RuntimeError(f"unexpected converted expert layout for {key}")
        converted_state[key] = (
            source.view(num_experts, d_model, hidden_size)
            .transpose(1, 2)
            .contiguous()
            .view_as(target)
        )
        repaired.append(key)
    return repaired


def run_revision(args: argparse.Namespace) -> Path:
    device = choose_device(args.device)
    noncuda_fallback = install_noncuda_moe_inference_fallback(device)
    revision = resolve_revision(args.repo, args.revision)
    layers = sorted({int(layer) for layer in args.layers.split(",")})
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = (
        f"{args.condition_set}-layers{'_'.join(str(layer) for layer in layers)}"
        f"-s{args.sequence_length}-c{args.calibration_sequences_per_label}"
        f"-e{args.eval_sequences_per_label}"
    )
    output_path = output_dir / f"{_safe_name(revision)}--{profile}.json"
    requested_manifest = {
        "schema_version": SCHEMA_VERSION,
        "repo": args.repo,
        "revision": revision,
        "layers": layers,
        "condition_set": args.condition_set,
        "sequence_length": args.sequence_length,
        "calibration_per_label": args.calibration_sequences_per_label,
        "evaluation_per_label": args.eval_sequences_per_label,
        "candidate_count": args.candidate_count,
        "power_iterations": args.power_iterations,
        "seed": args.seed,
        "git_commit": _git_commit(),
    }
    manifest_hash = hashlib.sha256(
        json.dumps(requested_manifest, sort_keys=True).encode()
    ).hexdigest()
    if output_path.exists() and not args.force:
        existing = json.loads(output_path.read_text())
        if existing.get("complete") and existing.get("manifest_hash") == manifest_hash:
            print(f"already complete: {output_path}")
            return output_path

    print(f"loading {args.repo}@{revision} on {device}", flush=True)
    model, tokenizer, snapshot = load_native_model(
        repo=args.repo,
        revision=revision,
        device=device,
        local_files_only=args.local_files_only,
    )
    if args.local_smoke:
        calibration_samples, eval_samples = local_smoke_samples(tokenizer, args.sequence_length)
    else:
        specs = parse_dataset_specs(args.dataset_spec)
        calibration_samples, eval_samples = numpy_validation_samples(
            specs,
            sequence_length=args.sequence_length,
            calibration_per_label=args.calibration_sequences_per_label,
            evaluation_per_label=args.eval_sequences_per_label,
        )

    modules = target_modules(model, layers)
    bundles: dict[int, ControlBundle] = {}
    moment_hooks: dict[int, ActivationMomentHook] = {}
    for layer, (router, mlp) in modules.items():
        print(f"constructing layer {layer} controls", flush=True)
        bundle = build_control_bundle(
            router,
            mlp,
            candidate_count=args.candidate_count,
            power_iterations=args.power_iterations,
            seed=args.seed + layer,
        )
        bundles[layer] = bundle
        moment_hooks[layer] = ActivationMomentHook(mlp, bundle)

    print("calibrating activation energy", flush=True)
    run_calibration(model, calibration_samples, moment_hooks, device=device)
    for layer in layers:
        select_calibrated_controls(bundles[layer], moment_hooks[layer])

    baseline_route_hooks = {
        layer: RouteFingerprintHook(router) for layer, (router, _mlp) in modules.items()
    }
    strict = args.condition_set == "smoke"
    print("evaluating baseline", flush=True)
    baseline, baseline_routes, baseline_logits = evaluate_samples(
        model,
        eval_samples,
        device=device,
        route_hooks=baseline_route_hooks,
        keep_logits=strict,
    )
    if strict and (
        baseline["direct_ce_max_abs_error"] > 2e-5
        or baseline["evaluator_direct_max_abs_error"] > 2e-5
    ):
        raise RuntimeError(
            "native evaluator CE does not match direct token CE: "
            f"{baseline['direct_ce_max_abs_error']=}, "
            f"{baseline['evaluator_direct_max_abs_error']=}"
        )
    repeat_logit_floor = 0.0
    repeat_weight_floor = 0.0
    repeat_route_comparison: dict[str, Any] | None = None
    if strict:
        print("measuring no-hook repeatability floor", flush=True)
        _repeat, repeat_routes, repeat_logits = evaluate_samples(
            model,
            eval_samples,
            device=device,
            route_hooks={
                layer: RouteFingerprintHook(router) for layer, (router, _mlp) in modules.items()
            },
            keep_logits=True,
        )
        assert baseline_logits is not None and repeat_logits is not None
        repeat_logit_floor = max(
            float((left.float() - right.float()).abs().max())
            for left, right in zip(baseline_logits, repeat_logits)
        )
        repeat_route_comparison = compare_route_captures(baseline_routes, repeat_routes)
        if not repeat_route_comparison["ids_and_counts_identical"]:
            raise RuntimeError("no-hook repeated evaluation changed target expert assignments")
        repeat_weight_floor = repeat_route_comparison["weights_max_abs_error"]
    if noncuda_fallback:
        # The pure-PyTorch scatter uses BF16 index_add. Its parallel accumulation
        # order is not stable on MPS, so use a BF16-scale floor while still
        # requiring exact discrete assignments.
        logit_tolerance = max(
            2.0 * repeat_logit_floor + 1e-7,
            4.0 * torch.finfo(torch.bfloat16).eps,
        )
        route_weight_tolerance = max(
            2.0 * repeat_weight_floor + 1e-7,
            2.0 * torch.finfo(torch.bfloat16).eps,
        )
    else:
        logit_tolerance = max(2.0 * repeat_logit_floor + 1e-7, 1e-6)
        route_weight_tolerance = max(2.0 * repeat_weight_floor + 1e-7, 2e-6)
    result: dict[str, Any] = {
        **requested_manifest,
        "manifest_hash": manifest_hash,
        "device": str(device),
        "noncuda_moe_inference_fallback": noncuda_fallback,
        "snapshot": str(snapshot),
        "sample_ids": {
            "calibration": [sample.sample_id for sample in calibration_samples],
            "evaluation": [sample.sample_id for sample in eval_samples],
        },
        "baseline": baseline,
        "baseline_route_fingerprints": serialize_route_captures(baseline_routes),
        "layers": {},
        "invariants": {
            "no_hook_repeat_logits_max_abs_error": repeat_logit_floor,
            "no_hook_repeat_routes": repeat_route_comparison,
            "logit_tolerance": logit_tolerance,
            "route_weight_tolerance": route_weight_tolerance,
        },
        "complete": False,
    }

    for layer in layers:
        router, mlp = modules[layer]
        initial_pre_hook_count = len(mlp._forward_pre_hooks)
        initial_post_hook_count = len(mlp._forward_hooks)
        bundle = bundles[layer]
        layer_record: dict[str, Any] = {
            "controls": {
                "candidate_eigenvalues": bundle.candidate_eigenvalues,
                "router_quadratic_energy": bundle.router_quadratic_energy,
                "calibration": bundle.calibration,
                "candidate_router_cosine_max": torch.einsum(
                    "ekd,ed->ek", bundle.candidates, bundle.router
                )
                .abs()
                .max(),
            },
            "conditions": {},
        }
        for condition in condition_set(args.condition_set):
            print(f"layer {layer}: {condition.name}", flush=True)
            projection = GroupedProjectionHook(
                mlp,
                bundle.direction(condition.direction_name),
                condition,
                validate_manual=strict,
            )
            route = RouteFingerprintHook(router)
            with projection:
                record, routes, logits = evaluate_samples(
                    model,
                    eval_samples,
                    device=device,
                    route_hooks={layer: route},
                    keep_logits=strict,
                )
            record["route_fingerprints"] = serialize_route_captures(routes)[str(layer)]
            record["projection"] = projection.diagnostics()
            route_comparison = compare_route_captures(
                {str(layer): baseline_routes[str(layer)]}, routes
            )
            record["target_route_comparison"] = route_comparison
            record["target_routes_identical"] = (
                route_comparison["ids_and_counts_identical"]
                and route_comparison["weights_max_abs_error"] <= route_weight_tolerance
            )
            if not record["target_routes_identical"]:
                raise RuntimeError(
                    f"target-layer routing changed under condition {condition.name} at layer {layer}"
                )
            if strict and condition.name == "identity_alpha0":
                assert baseline_logits is not None and logits is not None
                max_error = max(
                    float((left.float() - right.float()).abs().max())
                    for left, right in zip(baseline_logits, logits)
                )
                record["baseline_logits_max_abs_error"] = max_error
                if max_error > logit_tolerance:
                    raise RuntimeError(
                        f"alpha=0 exceeded the no-hook numerical floor: "
                        f"max error {max_error}, tolerance {logit_tolerance}"
                    )
            layer_record["conditions"][condition.name] = record
        layer_record["hook_counts_restored"] = {
            "pre": len(mlp._forward_pre_hooks) == initial_pre_hook_count,
            "post": len(mlp._forward_hooks) == initial_post_hook_count,
        }
        if not all(layer_record["hook_counts_restored"].values()):
            raise RuntimeError(f"layer {layer} intervention hooks leaked")
        add_layer_statistics(layer_record, baseline)
        result["layers"][str(layer)] = layer_record

    if strict:
        restored, restored_routes, restored_logits = evaluate_samples(
            model,
            eval_samples,
            device=device,
            route_hooks={
                layer: RouteFingerprintHook(router) for layer, (router, _mlp) in modules.items()
            },
            keep_logits=True,
        )
        assert restored_logits is not None and baseline_logits is not None
        restoration_error = max(
            float((left.float() - right.float()).abs().max())
            for left, right in zip(baseline_logits, restored_logits)
        )
        restored_route_comparison = compare_route_captures(baseline_routes, restored_routes)
        result["invariants"].update(
            {
                "hooks_removed_logits_max_abs_error": restoration_error,
                "hooks_removed_routes": restored_route_comparison,
                "restored_mean_ce": restored["mean_ce"],
            }
        )
        if (
            restoration_error > logit_tolerance
            or not restored_route_comparison["ids_and_counts_identical"]
            or restored_route_comparison["weights_max_abs_error"] > route_weight_tolerance
        ):
            raise RuntimeError(
                "removing hooks did not restore baseline within numerical floor: "
                f"logit_error={restoration_error}, logit_tolerance={logit_tolerance}, "
                f"routes={restored_route_comparison}, "
                f"route_weight_tolerance={route_weight_tolerance}"
            )

    result["complete"] = True
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(f"wrote {output_path}", flush=True)
    return output_path


def load_completed_results(
    output_dir: Path, *, condition_set_name: str | None = None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if (
                record.get("complete")
                and "revision" in record
                and "layers" in record
                and (
                    condition_set_name is None or record.get("condition_set") == condition_set_name
                )
            ):
                results.append(record)
        except (json.JSONDecodeError, ValueError):
            continue
    return results


def gate_pilot(output_dir: Path, layer: int) -> dict[str, Any]:
    pilot_records = load_completed_results(output_dir, condition_set_name="pilot")
    results = {parse_step(record["revision"]): record for record in pilot_records}
    missing = [step for step in PILOT_STEPS if step not in results]
    if missing:
        raise RuntimeError(f"pilot results are incomplete; missing steps {missing}")
    late: dict[str, Any] = {}
    passed = 0
    for step in LATE_GATE_STEPS:
        record = results[step]["layers"][str(layer)]
        d_record = record["D_alpha1"]
        lower, upper = d_record["ci95"]
        positive = lower > 0
        passed += int(positive)
        late[str(step)] = {
            "D_mean": d_record["mean"],
            "ci95": [lower, upper],
            "positive": positive,
        }
    final = results[max(PILOT_STEPS)]["layers"][str(layer)]["conditions"]
    delta_half = final["router_alpha0.5"]["delta_ce_mean"]
    delta_full = final["router_alpha1"]["delta_ce_mean"]
    monotonic = 0.0 <= delta_half <= delta_full
    decision = passed >= 3 and monotonic
    gate = {
        "layer": layer,
        "late_milestones": late,
        "positive_late_milestones": passed,
        "dose_response": {
            "alpha0": 0.0,
            "alpha0.5": delta_half,
            "alpha1": delta_full,
            "monotonic": monotonic,
        },
        "continue_full_sweep": decision,
    }
    path = output_dir / "pilot-gate.json"
    path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return gate


def summarize_results(output_dir: Path) -> Path:
    results = load_completed_results(output_dir)
    summary_path = output_dir / "summary.csv"
    rows: list[dict[str, Any]] = []

    def summary_step(record: Mapping[str, Any]) -> int:
        try:
            return parse_step(record["revision"])
        except ValueError:
            return -1

    for result in sorted(
        results,
        key=lambda record: (
            summary_step(record),
            record["condition_set"],
            tuple(record["layers"].keys()),
        ),
    ):
        step = summary_step(result)
        baseline = result["baseline"]["mean_ce"]
        for layer, layer_record in result["layers"].items():
            for name, condition in layer_record["conditions"].items():
                rows.append(
                    {
                        "step": step,
                        "revision": result["revision"],
                        "condition_set": result["condition_set"],
                        "sequence_length": result["sequence_length"],
                        "layer": int(layer),
                        "condition": name,
                        "baseline_ce": baseline,
                        "condition_ce": condition["mean_ce"],
                        "delta_ce": condition["delta_ce_mean"],
                        "ci95_lower": condition["delta_ce_ci95"][0],
                        "ci95_upper": condition["delta_ce_ci95"][1],
                    }
                )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "revision",
                "condition_set",
                "sequence_length",
                "layer",
                "condition",
                "baseline_ce",
                "condition_ce",
                "delta_ce",
                "ci95_lower",
                "ci95_upper",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        middle_layer = 8
        time_rows = [
            row
            for row in rows
            if row["layer"] == middle_layer
            and row["condition"] in {"router_alpha1", "matched_alpha1"}
            and row["condition_set"] == "full"
        ]
        if not time_rows:
            time_rows = [
                row
                for row in rows
                if row["layer"] == middle_layer
                and row["condition"] in {"router_alpha1", "matched_alpha1"}
                and row["condition_set"] in {"pilot", "smoke"}
            ]
        if time_rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            for condition in ("router_alpha1", "matched_alpha1"):
                subset = sorted(
                    (row for row in time_rows if row["condition"] == condition),
                    key=lambda row: row["step"],
                )
                ax.plot(
                    [row["step"] for row in subset],
                    [row["delta_ce"] for row in subset],
                    marker="o",
                    label=condition,
                )
            if all(row["step"] > 0 for row in time_rows):
                ax.set_xscale("log")
            ax.set_xlabel("checkpoint step")
            ax.set_ylabel("paired ΔCE from baseline")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "router-axis-time-course.png", dpi=180)
            plt.close(fig)

        final_records = sorted(
            results,
            key=lambda record: (
                summary_step(record),
                len(record["layers"]),
                record["sequence_length"],
            ),
        )
        if final_records:
            final_record = final_records[-1]
            layer_rows = [
                row
                for row in rows
                if row["revision"] == final_record["revision"]
                and row["condition_set"] == final_record["condition_set"]
                and row["sequence_length"] == final_record["sequence_length"]
                and row["condition"] in {"router_alpha1", "matched_alpha1"}
            ]
            if layer_rows:
                fig, ax = plt.subplots(figsize=(8, 4))
                for condition in ("router_alpha1", "matched_alpha1"):
                    subset = sorted(
                        (row for row in layer_rows if row["condition"] == condition),
                        key=lambda row: row["layer"],
                    )
                    ax.plot(
                        [row["layer"] for row in subset],
                        [row["delta_ce"] for row in subset],
                        marker="o",
                        label=condition,
                    )
                ax.set_xlabel("layer")
                ax.set_ylabel("paired ΔCE from baseline")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / "router-axis-across-layers.png", dpi=180)
                plt.close(fig)

            final_layer = final_record["layers"].get(str(middle_layer))
            if final_layer is not None:
                conditions = final_layer["conditions"]
                if all(
                    key in conditions
                    for key in (
                        "router_alpha0.5",
                        "router_alpha1",
                        "matched_alpha0.5",
                        "matched_alpha1",
                    )
                ):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(
                        [0, 0.5, 1],
                        [
                            0,
                            conditions["router_alpha0.5"]["delta_ce_mean"],
                            conditions["router_alpha1"]["delta_ce_mean"],
                        ],
                        marker="o",
                        label="router axis",
                    )
                    ax.plot(
                        [0, 0.5, 1],
                        [
                            0,
                            conditions["matched_alpha0.5"]["delta_ce_mean"],
                            conditions["matched_alpha1"]["delta_ce_mean"],
                        ],
                        marker="o",
                        label="matched orthogonal",
                    )
                    ax.set_xlabel("α")
                    ax.set_ylabel("paired ΔCE from baseline")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output_dir / "router-axis-dose-response.png", dpi=180)
                    plt.close(fig)

            energy_layers: list[int] = []
            energy_ratios: list[list[float]] = []
            for layer, layer_record in sorted(
                final_record["layers"].items(), key=lambda item: int(item[0])
            ):
                ratios = layer_record["controls"]["calibration"]["matched_eta_ratio"]
                energy_layers.append(int(layer))
                energy_ratios.append(ratios)
            if energy_ratios:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot(energy_ratios, tick_labels=energy_layers, showfliers=False)
                ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
                ax.set_xlabel("layer")
                ax.set_ylabel("matched control η / router-axis η")
                fig.tight_layout()
                fig.savefig(output_dir / "router-axis-energy-matching.png", dpi=180)
                plt.close(fig)
    except ImportError:
        pass
    print(f"wrote {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("run", "list-revisions", "gate", "summarize"), default="run"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layers", default="8")
    parser.add_argument(
        "--condition-set", choices=("smoke", "pilot", "full", "final"), default="smoke"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="output/router-axis-causal-eval")
    parser.add_argument("--dataset-spec", action="append", default=[])
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--calibration-sequences-per-label", type=int, default=4)
    parser.add_argument("--eval-sequences-per-label", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12_987)
    parser.add_argument("--gate-layer", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "list-revisions":
        for revision in public_step_revisions(args.repo):
            print(revision)
        return 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.mode == "gate":
        gate = gate_pilot(output_dir, args.gate_layer)
        return 0 if gate["continue_full_sweep"] else 2
    if args.mode == "summarize":
        summarize_results(output_dir)
        return 0
    run_revision(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
