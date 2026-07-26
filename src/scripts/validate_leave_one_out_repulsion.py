#!/usr/bin/env python3
"""Compare the old attractive and fixed repulsive centroid updates."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from olmo_core.nn.moe.router import (
    MoECentroidRouter,
    MoELeaveOneOutCentroidRouter,
    MoERouterGatingFunction,
)


class TinyMoEGraph(nn.Module):
    """Small graph with the same matrices surrounding a top-k SwiGLU MoE."""

    def __init__(self, *, spherical: bool):
        super().__init__()
        d_model, hidden, num_experts, top_k = 8, 12, 4, 2
        self.pre = nn.Linear(d_model, d_model, bias=False)
        self.router = MoELeaveOneOutCentroidRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            gating_function=MoERouterGatingFunction.identity,
            centroid_alpha=0.8,
            centroid_spherical=spherical,
        )
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.post = nn.Linear(d_model, 3, bias=False)
        nn.init.normal_(self.w1, std=0.1)
        nn.init.normal_(self.w3, std=0.1)
        nn.init.normal_(self.w2, std=0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        weights, indices, _, _ = self.router(h)
        up = torch.einsum("btd,ehd->bteh", h, self.w1)
        gate = torch.einsum("btd,ehd->bteh", h, self.w3)
        expert_hidden = F.silu(up) * gate
        expert_outputs = torch.einsum("bteh,edh->bted", expert_hidden, self.w2)
        selected = torch.gather(
            expert_outputs,
            2,
            indices.unsqueeze(-1).expand(*indices.shape, h.shape[-1]),
        )
        mixed = (weights.unsqueeze(-1) * selected).sum(dim=2)
        return self.post(h + mixed), indices


def _make_models() -> Dict[str, TinyMoEGraph]:
    torch.manual_seed(6198)
    base = TinyMoEGraph(spherical=False)
    state = deepcopy(base.state_dict())
    models = {
        "old_attractive": TinyMoEGraph(spherical=False),
        "repulsive_raw": TinyMoEGraph(spherical=False),
        "repulsive_spherical": TinyMoEGraph(spherical=True),
    }
    for model in models.values():
        model.load_state_dict(state)
        model.train()
    return models


def _parameter_norms(model: nn.Module, attr: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        tensor = getattr(parameter, attr)
        values[name] = float(tensor.float().norm()) if tensor is not None else 0.0
    return values


def _run_adamw_step(
    model: TinyMoEGraph,
    x: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad()
    output, indices = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    grad_norms = _parameter_norms(model, "grad")
    optimizer.step()

    update_norms = {
        name: float((parameter.detach() - before[name]).float().norm())
        for name, parameter in model.named_parameters()
    }
    exp_avg_norms = {
        name: float(optimizer.state[parameter]["exp_avg"].float().norm())
        for name, parameter in model.named_parameters()
    }
    exp_avg_sq_norms = {
        name: float(optimizer.state[parameter]["exp_avg_sq"].float().norm())
        for name, parameter in model.named_parameters()
    }
    return {
        "loss": float(loss.detach()),
        "assignments": indices.detach().flatten().tolist(),
        "grad_norms": grad_norms,
        "adamw_update_norms": update_norms,
        "adamw_exp_avg_norms": exp_avg_norms,
        "adamw_exp_avg_sq_norms": exp_avg_sq_norms,
        "router_parameter_count": sum(p.numel() for p in model.router.parameters()),
        "router_entries_in_adamw_state": sum(
            parameter in optimizer.state for parameter in model.router.parameters()
        ),
    }


def main() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    adapt_x = torch.randn(3, 5, 8, generator=generator)
    train_x = torch.randn(2, 4, 8, generator=generator)
    target = torch.randn(2, 4, 3, generator=generator)

    # Before any centroid M-step, old and fixed routing have an identical
    # forward/backward graph. This isolates all later AdamW differences to the
    # changed centroid state, not to an accidental same-step graph change.
    pre_models = _make_models()
    pre_gradients: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in ("old_attractive", "repulsive_spherical"):
        output, _ = pre_models[name](train_x)
        F.mse_loss(output, target).backward()
        pre_gradients[name] = {
            parameter_name: parameter.grad.detach().clone()
            for parameter_name, parameter in pre_models[name].named_parameters()
        }
    pre_update_max_grad_abs_diff = max(
        float(
            (
                pre_gradients["old_attractive"][parameter_name]
                - pre_gradients["repulsive_spherical"][parameter_name]
            )
            .abs()
            .max()
        )
        for parameter_name in pre_gradients["old_attractive"]
    )

    models = _make_models()
    centroid_metrics: Dict[str, object] = {}
    for name, model in models.items():
        with torch.no_grad():
            h = model.pre(adapt_x)
        before = model.router._centroid.detach().clone()
        _, indices, _, _ = model.router(h)

        if name == "old_attractive":
            # Bypass the fixed override to reproduce the previous implementation:
            # the complement mean is used with a positive sign.
            MoECentroidRouter.post_batch(model.router)
        else:
            model.router.post_batch()

        after = model.router._centroid.detach().clone()
        scores_before = torch.einsum("btd,ed->bte", h, before)
        scores_after = torch.einsum("btd,ed->bte", h, after)
        selected = torch.zeros_like(scores_before, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        complement_score_delta = (scores_after - scores_before)[~selected]
        centroid_metrics[name] = {
            "centroid_norm_mean": float(after.norm(dim=-1).mean()),
            "centroid_norm_min": float(after.norm(dim=-1).min()),
            "centroid_norm_max": float(after.norm(dim=-1).max()),
            "centroid_update_norm_mean": float((after - before).norm(dim=-1).mean()),
            "mean_unselected_score_delta": float(complement_score_delta.mean()),
            "fraction_unselected_scores_decreased": float(
                (complement_score_delta < 0).float().mean()
            ),
        }

    adamw_metrics = {
        name: _run_adamw_step(model, train_x, target) for name, model in models.items()
    }
    report = {
        "pre_update_max_grad_abs_diff": pre_update_max_grad_abs_diff,
        "centroid_metrics_after_one_m_step": centroid_metrics,
        "next_batch_backprop_and_adamw": adamw_metrics,
    }

    def _all_finite(value: object) -> bool:
        if isinstance(value, dict):
            return all(_all_finite(item) for item in value.values())
        if isinstance(value, list):
            return all(_all_finite(item) for item in value)
        if isinstance(value, float):
            return torch.isfinite(torch.tensor(value)).item()
        return True

    assert pre_update_max_grad_abs_diff == 0.0
    assert centroid_metrics["old_attractive"]["mean_unselected_score_delta"] > 0  # type: ignore[index,operator]
    assert centroid_metrics["repulsive_raw"]["mean_unselected_score_delta"] < 0  # type: ignore[index,operator]
    assert centroid_metrics["repulsive_spherical"]["mean_unselected_score_delta"] < 0  # type: ignore[index,operator]
    assert all(metrics["router_parameter_count"] == 0 for metrics in adamw_metrics.values())
    assert all(metrics["router_entries_in_adamw_state"] == 0 for metrics in adamw_metrics.values())
    assert _all_finite(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
