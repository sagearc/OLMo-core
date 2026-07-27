import pytest
import torch

from olmo_core.nn.moe.router import (
    MoEBinaryLeaveOneOutLinearRouter,
    MoELinearRouter,
    MoERouterGatingFunction,
)
from olmo_core.testing import DEVICES


class _CaptureMixin:
    last_logits: torch.Tensor

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().get_expert_logits(x)  # type: ignore[misc]
        logits.retain_grad()
        self.last_logits = logits
        return logits


class _Regular(_CaptureMixin, MoELinearRouter):
    pass


class _LeaveOneOut(_CaptureMixin, MoEBinaryLeaveOneOutLinearRouter):
    pass


def _kwargs() -> dict:
    return {
        "d_model": 4,
        "num_experts": 2,
        "top_k": 1,
        "gating_function": MoERouterGatingFunction.sigmoid,
        "normalize_expert_weights": None,
        "bias_gamma": 1e-3,
    }


def _match(regular: _Regular, loo: _LeaveOneOut, device: torch.device) -> None:
    with torch.no_grad():
        weight = torch.tensor(
            [[0.08, -0.03, 0.05, 0.02], [-0.04, 0.07, -0.01, 0.06]],
            device=device,
        )
        regular.weight.copy_(weight.flatten())
        loo.weight.copy_(weight.flatten())
        assert regular.score_bias is not None and loo.score_bias is not None
        bias = torch.tensor([-0.01, 0.01], device=device)
        regular.score_bias.copy_(bias)
        loo.score_bias.copy_(bias)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [(1, 1), (1, 2), (2, 1), (2, 2)],
)
def test_binary_leave_one_out_forward_and_gradient(
    device: torch.device,
    batch_size: int,
    sequence_length: int,
):
    regular = _Regular(**_kwargs()).to(device)
    loo = _LeaveOneOut(**_kwargs()).to(device)
    _match(regular, loo, device)
    tokens = torch.tensor(
        [
            [[0.9, -0.2, 0.5, 0.1], [-0.3, 0.8, 0.2, -0.6]],
            [[0.4, 0.1, -0.7, 0.9], [-0.8, -0.3, 0.6, 0.2]],
        ],
        device=device,
    )
    x = tokens[:batch_size, :sequence_length]
    regular_weights, regular_indices, regular_load, _ = regular(x)
    loo_weights, loo_indices, loo_load, _ = loo(x)

    regular_scores = torch.sigmoid(regular.last_logits) + 1e-7
    loo_scores = 1.0 - torch.sigmoid(loo.last_logits).flip(dims=(-1,)) + 1e-7
    assert torch.all(regular_scores > 0)
    assert torch.all(loo_scores > 0)
    torch.testing.assert_close(
        loo_scores[..., 0] - loo_scores[..., 1],
        regular_scores[..., 0] - regular_scores[..., 1],
    )
    torch.testing.assert_close(loo_indices, regular_indices, rtol=0.0, atol=0.0)
    torch.testing.assert_close(loo_load, regular_load, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        regular_weights, regular_scores.gather(-1, regular_indices)
    )
    torch.testing.assert_close(loo_weights, loo_scores.gather(-1, loo_indices))

    upstream = torch.tensor(
        [[[0.7], [-0.4]], [[0.2], [0.9]]], device=device
    )[:batch_size, :sequence_length]
    (regular_weights * upstream).sum().backward()
    (loo_weights * upstream).sum().backward()
    assert regular.last_logits.grad is not None
    assert loo.last_logits.grad is not None
    selected = torch.zeros_like(loo.last_logits.grad, dtype=torch.bool)
    selected.scatter_(-1, loo_indices, True)
    torch.testing.assert_close(
        regular.last_logits.grad.masked_select(~selected),
        torch.zeros_like(regular.last_logits.grad.masked_select(~selected)),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        loo.last_logits.grad.masked_select(selected),
        torch.zeros_like(loo.last_logits.grad.masked_select(selected)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(regular.last_logits.grad.masked_select(selected).abs() > 0)
    assert torch.all(loo.last_logits.grad.masked_select(~selected).abs() > 0)


class _TwoLayer(torch.nn.Module):
    def __init__(self, router_type: type[MoELinearRouter], device: torch.device):
        super().__init__()
        self.routers = torch.nn.ModuleList(
            [router_type(**_kwargs()), router_type(**_kwargs())]
        )
        self.register_buffer(
            "expert_vectors",
            torch.tensor(
                [
                    [[0.2, -0.1, 0.3, 0.05], [-0.3, 0.4, 0.1, -0.2]],
                    [[-0.25, 0.3, 0.15, -0.1], [0.4, -0.2, 0.05, 0.35]],
                ],
                device=device,
            ),
        )

    def forward(self, x: torch.Tensor):
        traces = []
        for layer, router in enumerate(self.routers):
            weights, indices, _, _ = router(x)
            x = x + (weights.unsqueeze(-1) * self.expert_vectors[layer][indices]).sum(
                dim=-2
            )
            traces.append((router.last_logits, indices))
        return x.square().mean(), traces


@pytest.mark.parametrize("device", DEVICES)
def test_binary_leave_one_out_two_layer_selected_rows_stay_gradient_free(
    device: torch.device,
):
    model = _TwoLayer(_LeaveOneOut, device).to(device)
    loss, traces = model(
        torch.tensor(
            [[[0.9, -0.2, 0.5, 0.1], [-0.3, 0.8, 0.2, -0.6]]],
            device=device,
        )
    )
    loss.backward()
    for router, (logits, indices) in zip(model.routers, traces):
        assert logits.grad is not None
        selected = torch.zeros_like(logits.grad, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        torch.testing.assert_close(
            logits.grad.masked_select(selected),
            torch.zeros_like(logits.grad.masked_select(selected)),
            rtol=0.0,
            atol=0.0,
        )
        assert router.weight.grad is not None
        assert torch.isfinite(router.weight.grad).all()
