from copy import deepcopy

import pytest
import torch

from olmo_core.nn.moe.router import (
    MoECentroidRouter,
    MoELeaveOneOutCentroidRouter,
    MoELeaveOneOutLinearRouter,
    MoELinearRouter,
    MoEOrthogonalCentroidRouter,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoERouterType,
)
from olmo_core.testing import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "uniform_expert_assignment",
    [
        pytest.param(True, id="uniform"),
        pytest.param(False, id="computed"),
    ],
)
@pytest.mark.parametrize(
    "gating_function",
    [
        pytest.param(MoERouterGatingFunction.softmax, id="softmax"),
        pytest.param(MoERouterGatingFunction.sigmoid, id="sigmoid"),
    ],
)
def test_router(
    device: torch.device, uniform_expert_assignment: bool, gating_function: MoERouterGatingFunction
):
    router = MoELinearRouter(
        d_model=128,
        num_experts=4,
        jitter_eps=0.1,
        top_k=2,
        normalize_expert_weights=True,
        uniform_expert_assignment=uniform_expert_assignment,
        gating_function=gating_function,
    ).to(device)

    x = torch.randn((2, 4, 128), device=device)
    weights, indices, bz_per_expert, _ = router(x)

    assert weights.shape == (2, 4, 2)
    assert indices.shape == (2, 4, 2)
    assert bz_per_expert.shape == (4,)


@pytest.mark.parametrize("device", DEVICES)
def test_router_with_bias_gamma(device: torch.device):
    router1 = MoELinearRouter(
        d_model=128,
        num_experts=4,
        top_k=2,
        bias_gamma=0.001,
    ).to(device)
    router1.reset_parameters()

    assert router1.score_bias is not None
    assert router1.score_bias.nonzero().sum().item() == 0  # type: ignore
    assert router1.score_bias_batch_size_per_expert is not None
    assert router1.score_bias_batch_size_per_expert.nonzero().sum().item() == 0

    router2 = MoELinearRouter(
        d_model=128,
        num_experts=4,
        top_k=2,
    ).to(device)
    router2.reset_parameters()
    state_dict = router1.state_dict()
    del state_dict["score_bias"]
    router2.load_state_dict(state_dict)

    x = torch.randn((2, 4, 128), device=device)

    # At this point, the output should be exactly the same as it would be without a bias gamma.
    weights1, indices1, bz_per_expert1, _ = router1(x)
    weights2, indices2, bz_per_expert2, _ = router2(x)
    torch.testing.assert_close(weights1, weights2)
    torch.testing.assert_close(indices1, indices2)
    torch.testing.assert_close(bz_per_expert1, bz_per_expert2)

    assert router1.batch_size_per_expert.sum().item() == 8 * 2

    # Update the biases and check.
    router1.post_batch()
    assert router1.score_bias.nonzero().sum().item() > 0  # type: ignore
    assert router1.score_bias_batch_size_per_expert is not None
    assert router1.score_bias_batch_size_per_expert.nonzero().sum().item() == 0


def test_router_config_builds_router_variants():
    assert isinstance(MoERouterConfig().build(8, 4), MoELinearRouter)
    assert isinstance(
        MoERouterConfig(name=MoERouterType.leave_one_out).build(8, 4),
        MoELeaveOneOutLinearRouter,
    )
    assert isinstance(
        MoERouterConfig(name=MoERouterType.orthogonal_centroid).build(8, 4),
        MoEOrthogonalCentroidRouter,
    )
    assert isinstance(
        MoERouterConfig(name=MoERouterType.leave_one_out_centroid).build(8, 4),
        MoELeaveOneOutCentroidRouter,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_leave_one_out_linear_top1_excludes_routed_row_update(
    device: torch.device,
):
    # Deliberately omit L1 renormalization here: at top-1 it makes the sole
    # mixing weight identically one and therefore removes every router gradient.
    # This test isolates the leave-one-out dependency graph itself.
    router = MoELeaveOneOutLinearRouter(
        d_model=2,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.sigmoid,
    ).to(device)
    with torch.no_grad():
        router.weight.view(3, 2).copy_(
            torch.tensor(
                [[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                device=device,
            )
        )

    x = torch.tensor([[[1.0, 0.0]]], device=device)
    logits = router.get_expert_logits(x)
    torch.testing.assert_close(
        logits,
        torch.tensor([[[2.0, 0.0, -1.0]]], device=device),
        rtol=0.0,
        atol=0.0,
    )
    expert_weights, expert_indices, _, _ = router(x)
    assert expert_indices.item() == 0

    baseline = MoELinearRouter(
        d_model=2,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.sigmoid,
    ).to(device)
    with torch.no_grad():
        baseline.weight.copy_(router.weight)
    baseline_weights, baseline_indices, _, _ = baseline(x)
    torch.testing.assert_close(expert_weights, baseline_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert_indices, baseline_indices, rtol=0.0, atol=0.0)

    # Minimal one-token MoE: gather the routed expert's scalar response, mix it
    # by the router weight, and backpropagate a task loss through that output.
    expert_responses = torch.tensor([1.5, -0.75, 0.25], device=device)
    model_output = (expert_weights * expert_responses[expert_indices]).sum()
    loss = (model_output - 0.25).square()

    rows_before = router.weight.detach().view(3, 2).clone()
    loss.backward()
    assert router.weight.grad is not None
    grad = router.weight.grad.view(3, 2)
    torch.testing.assert_close(grad[0], torch.zeros_like(grad[0]), rtol=0.0, atol=0.0)
    assert grad[1].norm() > 0
    assert grad[2].norm() > 0

    torch.optim.SGD(router.parameters(), lr=0.1).step()
    rows_after = router.weight.detach().view(3, 2)
    torch.testing.assert_close(rows_after[0], rows_before[0], rtol=0.0, atol=0.0)
    assert not torch.equal(rows_after[1], rows_before[1])
    assert not torch.equal(rows_after[2], rows_before[2])


@pytest.mark.parametrize("device", DEVICES)
def test_leave_one_out_linear_top6_excludes_all_routed_row_updates(
    device: torch.device,
):
    router = MoELeaveOneOutLinearRouter(
        d_model=2,
        num_experts=8,
        top_k=6,
        gating_function=MoERouterGatingFunction.sigmoid,
        normalize_expert_weights=1.0,
    ).to(device)
    with torch.no_grad():
        router.weight.view(8, 2).copy_(
            torch.tensor(
                [
                    [4.0, 0.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 0.0],
                    [0.25, 0.0],
                    [-1.0, 0.0],
                    [-2.0, 0.0],
                ],
                device=device,
            )
        )

    x = torch.tensor([[[1.0, 0.0]]], device=device)
    expert_weights, expert_indices, _, _ = router(x)
    selected = expert_indices.flatten()
    torch.testing.assert_close(
        selected.sort().values,
        torch.arange(6, device=device),
        rtol=0.0,
        atol=0.0,
    )

    baseline = MoELinearRouter(
        d_model=2,
        num_experts=8,
        top_k=6,
        gating_function=MoERouterGatingFunction.sigmoid,
        normalize_expert_weights=1.0,
    ).to(device)
    with torch.no_grad():
        baseline.weight.copy_(router.weight)
    baseline_weights, baseline_indices, _, _ = baseline(x)
    torch.testing.assert_close(expert_weights, baseline_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert_indices, baseline_indices, rtol=0.0, atol=0.0)

    expert_responses = torch.arange(1, 9, dtype=x.dtype, device=device)
    model_output = (expert_weights * expert_responses[expert_indices]).sum()
    model_output.square().backward()
    assert router.weight.grad is not None
    grad = router.weight.grad.view(8, 2)
    torch.testing.assert_close(grad[:6], torch.zeros_like(grad[:6]), rtol=0.0, atol=0.0)
    assert grad[6].norm() > 0
    assert grad[7].norm() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_leave_one_out_centroid_top1_repels_unrouted_centroids(
    device: torch.device,
):
    router = MoELeaveOneOutCentroidRouter(
        d_model=2,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.identity,
        centroid_alpha=0.5,
    ).to(device)
    with torch.no_grad():
        router._centroid.copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                device=device,
            )
        )

    x = torch.tensor([[[2.0, 1.0]]], device=device)
    expert_weights, expert_indices, _, _ = router(x)
    assert expert_indices.item() == 0

    baseline = MoECentroidRouter(
        d_model=2,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.identity,
        centroid_alpha=0.5,
    ).to(device)
    with torch.no_grad():
        baseline._centroid.copy_(router._centroid)
    baseline_weights, baseline_indices, _, _ = baseline(x)
    torch.testing.assert_close(expert_weights, baseline_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert_indices, baseline_indices, rtol=0.0, atol=0.0)

    centroids_before = router._centroid.clone()

    router.post_batch()
    torch.testing.assert_close(router._centroid[0], centroids_before[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        router._centroid[1],
        torch.tensor([-1.0, 0.0], device=device),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        router._centroid[2],
        torch.tensor([-1.0, -1.0], device=device),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_leave_one_out_centroid_top6_repels_all_unrouted_centroids(
    device: torch.device,
):
    router = MoELeaveOneOutCentroidRouter(
        d_model=2,
        num_experts=8,
        top_k=6,
        gating_function=MoERouterGatingFunction.identity,
        centroid_alpha=0.5,
    ).to(device)
    with torch.no_grad():
        router._centroid.copy_(
            torch.tensor(
                [
                    [4.0, 0.0],
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 0.0],
                    [0.25, 0.0],
                    [-1.0, 0.0],
                    [-2.0, 0.0],
                ],
                device=device,
            )
        )

    x = torch.tensor([[[1.0, 0.0]]], device=device)
    expert_weights, expert_indices, _, _ = router(x)
    selected = expert_indices.flatten()
    torch.testing.assert_close(
        selected.sort().values,
        torch.arange(6, device=device),
        rtol=0.0,
        atol=0.0,
    )

    baseline = MoECentroidRouter(
        d_model=2,
        num_experts=8,
        top_k=6,
        gating_function=MoERouterGatingFunction.identity,
        centroid_alpha=0.5,
    ).to(device)
    with torch.no_grad():
        baseline._centroid.copy_(router._centroid)
    baseline_weights, baseline_indices, _, _ = baseline(x)
    torch.testing.assert_close(expert_weights, baseline_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expert_indices, baseline_indices, rtol=0.0, atol=0.0)

    centroids_before = router._centroid.clone()
    router.post_batch()
    torch.testing.assert_close(
        router._centroid[:6],
        centroids_before[:6],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        router._centroid[6],
        torch.tensor([-1.0, 0.0], device=device),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        router._centroid[7],
        torch.tensor([-1.5, 0.0], device=device),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_leave_one_out_centroid_spherical_repulsion_preserves_magnitude(
    device: torch.device,
):
    router = MoELeaveOneOutCentroidRouter(
        d_model=3,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.identity,
        centroid_alpha=0.5,
        centroid_spherical=True,
    ).to(device)
    with torch.no_grad():
        router._centroid.copy_(torch.eye(3, device=device))

    x = torch.tensor([[[2.0, 1.0, 0.0]]], device=device)
    _, expert_indices, _, _ = router(x)
    assert expert_indices.item() == 0
    selected_before = router._centroid[0].clone()

    router.post_batch()

    torch.testing.assert_close(
        router._centroid.norm(dim=-1),
        torch.ones(3, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        router._centroid[0],
        selected_before,
        rtol=0.0,
        atol=0.0,
    )
    assert (x.flatten() @ router._centroid[1]) < (x.flatten() @ torch.eye(3, device=device)[1])
    assert (x.flatten() @ router._centroid[2]) < (x.flatten() @ torch.eye(3, device=device)[2])


def test_leave_one_out_centroid_is_not_an_adamw_parameter():
    router = MoELeaveOneOutCentroidRouter(
        d_model=4,
        num_experts=3,
        top_k=1,
        gating_function=MoERouterGatingFunction.identity,
        centroid_spherical=True,
    )
    assert list(router.named_parameters()) == []
    assert "_centroid" in dict(router.named_buffers())

    adjacent = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(adjacent.parameters(), lr=1e-3, weight_decay=0.1)
    optimizer.zero_grad()
    adjacent(torch.ones(1, 4)).square().mean().backward()
    optimizer.step()

    assert set(optimizer.state) == {adjacent.weight}
    assert router._centroid not in optimizer.state


@pytest.mark.parametrize("device", DEVICES)
def test_orthogonal_centroid_projection_is_matched_and_post_optim(device: torch.device):
    router = MoEOrthogonalCentroidRouter(
        d_model=16,
        num_experts=4,
        top_k=4,
        gating_function=MoERouterGatingFunction.sigmoid,
        normalize_expert_weights=1.0,
        centroid_alpha=0.5,
    ).to(device)
    router.train()
    x = torch.randn((2, 8, 16), device=device)
    weight_before = router.weight.detach().clone()
    router(x)
    router.post_batch()
    torch.testing.assert_close(router.weight, weight_before, rtol=0.0, atol=0.0)

    rows_before = router.weight.detach().view(4, 16).float().clone()
    norms_before = rows_before.norm(dim=-1)
    router.post_optim_step()
    rows_after = router.weight.detach().view(4, 16).float()
    centroid_hat = torch.nn.functional.normalize(router._routed_centroid, dim=-1)
    cosine_after = (torch.nn.functional.normalize(rows_after, dim=-1) * centroid_hat).sum(dim=-1)
    torch.testing.assert_close(rows_after.norm(dim=-1), norms_before, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cosine_after, torch.zeros_like(cosine_after), rtol=0.0, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_orthogonal_centroid_sham_matches_angle_and_norm(device: torch.device):
    common = dict(
        d_model=16,
        num_experts=4,
        top_k=4,
        gating_function=MoERouterGatingFunction.sigmoid,
        normalize_expert_weights=1.0,
        centroid_alpha=0.5,
        orthogonal_sham_seed=17,
    )
    matched = MoEOrthogonalCentroidRouter(**common).to(device)
    sham = MoEOrthogonalCentroidRouter(**common, orthogonal_sham=True).to(device)
    with torch.no_grad():
        sham.weight.copy_(matched.weight)
        centroid = torch.randn((4, 16), device=device)
        matched._routed_centroid.copy_(centroid)
        sham._routed_centroid.copy_(centroid)
        matched._routed_centroid_step.add_(1)
        sham._routed_centroid_step.add_(1)
    matched._routed_centroid_step_py = 1
    sham._routed_centroid_step_py = 1

    rows_before = matched.weight.detach().view(4, 16).float().clone()
    norms_before = rows_before.norm(dim=-1)
    matched.post_optim_step()
    sham.post_optim_step()
    matched_rows = matched.weight.detach().view(4, 16).float()
    sham_rows = sham.weight.detach().view(4, 16).float()

    def angle(rows_after: torch.Tensor) -> torch.Tensor:
        before_hat = torch.nn.functional.normalize(rows_before, dim=-1)
        after_hat = torch.nn.functional.normalize(rows_after, dim=-1)
        return torch.acos((before_hat * after_hat).sum(dim=-1).clamp(-1.0, 1.0))

    torch.testing.assert_close(angle(sham_rows), angle(matched_rows), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(sham_rows.norm(dim=-1), norms_before, rtol=2e-5, atol=2e-6)
    centroid_hat = torch.nn.functional.normalize(sham._routed_centroid, dim=-1)
    sham_matched_cosine = (torch.nn.functional.normalize(sham_rows, dim=-1) * centroid_hat).sum(
        dim=-1
    )
    assert sham_matched_cosine.abs().mean() > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_orthogonal_centroid_dry_run_does_not_commit(device: torch.device):
    router = MoEOrthogonalCentroidRouter(d_model=16, num_experts=4, top_k=2).to(device)
    router.train()
    router(torch.randn((2, 8, 16), device=device))
    router.post_batch(dry_run=True)
    assert router._routed_centroid_step_py == 0
    assert router._routed_centroid.abs().sum() == 0
    assert router._centroid_sum_accum is None
    assert router._centroid_count_accum is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sham", [False, True], ids=["matched", "sham"])
def test_orthogonal_centroid_state_dict_resume(device: torch.device, sham: bool):
    kwargs = dict(
        d_model=16,
        num_experts=4,
        top_k=2,
        centroid_alpha=0.5,
        orthogonal_sham=sham,
        orthogonal_sham_seed=23,
    )
    router1 = MoEOrthogonalCentroidRouter(**kwargs).to(device)
    router1.train()
    optim1 = torch.optim.AdamW(router1.parameters(), lr=1e-3)

    def training_step(
        router: MoEOrthogonalCentroidRouter,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        expert_weights, _, _, _ = router(x)
        expert_weights.float().square().mean().backward()
        router.post_batch(lr=1e-3)
        optimizer.step()
        router.post_optim_step()

    torch.manual_seed(19)
    training_step(router1, optim1, torch.randn((2, 8, 16), device=device))

    router2 = MoEOrthogonalCentroidRouter(**kwargs).to(device)
    router2.load_state_dict(router1.state_dict())
    router2.train()
    optim2 = torch.optim.AdamW(router2.parameters(), lr=1e-3)
    optim2.load_state_dict(deepcopy(optim1.state_dict()))
    assert router2._routed_centroid_step_py == router1._routed_centroid_step_py == 1
    assert router2._centroid_sum_accum is None
    assert router2._centroid_count_accum is None

    next_x = torch.randn((2, 8, 16), device=device)
    training_step(router1, optim1, next_x)
    training_step(router2, optim2, next_x)
    torch.testing.assert_close(router2.weight, router1.weight)
    torch.testing.assert_close(router2._routed_centroid, router1._routed_centroid)
    assert router2._routed_centroid_step_py == router1._routed_centroid_step_py == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "gating_function",
    [
        pytest.param(MoERouterGatingFunction.softmax, id="softmax"),
        pytest.param(MoERouterGatingFunction.sigmoid, id="sigmoid"),
    ],
)
def test_router_with_ema_zscore(device: torch.device, gating_function: MoERouterGatingFunction):
    router = MoELinearRouter(
        d_model=128,
        num_experts=4,
        top_k=2,
        gating_function=gating_function,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.9,
    ).to(device)
    router.train()

    # EMA buffers eagerly initialized to (0, 0); step_count starts at 0 so the first
    # forward bypasses normalization (no stats yet) — Adam-style cold-start.
    assert router._ema_mean is not None
    assert router._ema_var is not None
    assert router._ema_step_count is not None
    assert router._ema_mean.device.type == device.type
    assert router._ema_var.device.type == device.type
    assert router._ema_mean.abs().sum().item() == 0
    assert router._ema_var.abs().sum().item() == 0
    assert int(router._ema_step_count.item()) == 0

    x = torch.randn((2, 4, 128), device=device)
    weights, indices, _, _ = router(x)
    assert weights.shape == (2, 4, 2)
    assert indices.shape == (2, 4, 2)

    # Forward only ACCUMULATES — EMA buffers themselves are unchanged until post_batch.
    assert router._ema_mean.abs().sum().item() == 0
    assert int(router._ema_step_count.item()) == 0
    assert router._ema_logit_sum_accum is not None

    # post_batch performs the per-step EMA update + step counter increment.
    router.post_batch()
    assert router._ema_mean.abs().sum().item() > 0
    assert router._ema_var.abs().sum().item() > 0
    assert int(router._ema_step_count.item()) == 1
    assert router._ema_logit_sum_accum is None  # accumulators reset
    snapshot_mean = router._ema_mean.clone()
    snapshot_sq = router._ema_var.clone()
    snapshot_step = int(router._ema_step_count.item())

    # In eval mode the EMA should NOT update (no accumulation, post_batch is a no-op).
    router.eval()
    router(torch.randn((2, 4, 128), device=device))
    router.post_batch()
    torch.testing.assert_close(router._ema_mean, snapshot_mean)
    torch.testing.assert_close(router._ema_var, snapshot_sq)
    assert int(router._ema_step_count.item()) == snapshot_step
    assert router._ema_logit_sum_accum is None


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_zscore_backward(device: torch.device):
    """In-place EMA updates must not corrupt autograd of the gating path."""
    router = MoELinearRouter(
        d_model=64,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
    ).to(device)
    router.train()

    x = torch.randn((2, 8, 64), device=device, requires_grad=True)
    weights, _, _, _ = router(x)
    weights.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_tracks_input_distribution(device: torch.device):
    """After many updates, EMA mean/std should approach the true logit distribution."""
    router = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.5,  # fast decay for testing
    ).to(device)
    router.train()

    torch.manual_seed(0)
    for _ in range(50):
        router(torch.randn((4, 16, 32), device=device))
        router.post_batch()

    # `_ema_var` tracks Var(logit) directly and is always non-negative by construction
    # (EMA of non-negative sample-variance observations).
    assert router._ema_var is not None
    assert (router._ema_var > 0).all(), f"variance must be positive, got {router._ema_var}"


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_zscore_compute_metrics(device: torch.device):
    num_experts = 4
    router = MoELinearRouter(
        d_model=32,
        num_experts=num_experts,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.5,
    ).to(device)
    router.train()

    # t=0: compute_metrics should emit zero-valued mean/std and ema step = 0.
    metrics = router.compute_metrics(reset=False)
    assert metrics["ema step"][0].item() == 0
    for i in range(num_experts):
        assert metrics[f"expert {i:02d}/ema mean"][0].item() == 0.0
        assert metrics[f"expert {i:02d}/ema std"][0].item() == 0.0

    # Advance the EMA past t=0.
    torch.manual_seed(0)
    for _ in range(5):
        router(torch.randn((4, 16, 32), device=device))
        router.post_batch()

    metrics = router.compute_metrics(reset=False)
    assert metrics["ema step"][0].item() == 5
    for i in range(num_experts):
        mean_val = metrics[f"expert {i:02d}/ema mean"][0]
        std_val = metrics[f"expert {i:02d}/ema std"][0]
        assert mean_val.ndim == 0 and torch.isfinite(mean_val)
        assert std_val.ndim == 0 and torch.isfinite(std_val)
        assert std_val.item() > 0.0

    # Bias-corrected values should match what `_apply_ema_zscore` would use.
    step = int(router._ema_step_count.item())
    bias_correction = 1.0 - (router.ema_zscore_alpha**step)
    expected_mean = router._ema_mean / bias_correction
    expected_std = (router._ema_var / bias_correction).clamp(min=1e-4).sqrt()
    for i in range(num_experts):
        assert torch.allclose(metrics[f"expert {i:02d}/ema mean"][0], expected_mean[i], atol=1e-6)
        assert torch.allclose(metrics[f"expert {i:02d}/ema std"][0], expected_std[i], atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_router_with_seq_aux_loss(device: torch.device):
    router = MoELinearRouter(
        d_model=64,
        num_experts=8,
        top_k=2,
        gating_function=MoERouterGatingFunction.sigmoid,
        seq_aux_loss_weight=1e-4,
    ).to(device)
    router.train()

    x = torch.randn((2, 8, 64), device=device, requires_grad=True)
    _, _, _, aux_loss = router(x)

    # The seq aux loss should be attached to aux_loss, scaled by the weight, and finite.
    assert aux_loss is not None
    assert torch.isfinite(aux_loss)

    # Backward should flow through to x via the routing scores.
    aux_loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Metric should accumulate the unscaled loss.
    metrics = router.compute_metrics(reset=False)
    assert "seq aux loss" in metrics
    assert "seq aux loss unscaled" in metrics
    unscaled = metrics["seq aux loss unscaled"][0]
    assert unscaled.item() > 0

    # reset_metrics zeroes the accumulator.
    router.reset_metrics()
    assert router.seq_aux_loss is not None
    assert router.seq_aux_loss.item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_seq_aux_loss_microbatch_invariance(device: torch.device):
    """
    Sum of seq-aux loss across microbatches equals single-batch loss when ``loss_div_factor``
    is the total batch tokens. This guarantees the per-step gradient scale is independent
    of the microbatch chunking strategy.
    """
    router = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.sigmoid,
        seq_aux_loss_weight=1.0,
    ).to(device)
    router.train()

    torch.manual_seed(0)
    full = torch.randn((8, 4, 32), device=device)
    total_tokens = float(full.shape[0] * full.shape[1])

    _, _, _, full_aux = router(full, loss_div_factor=total_tokens)
    full_value = float(full_aux)

    # Same input split into 4 microbatches of size 2.
    router.reset_metrics()
    summed = 0.0
    for chunk in full.chunk(4, dim=0):
        _, _, _, mb_aux = router(chunk, loss_div_factor=total_tokens)
        summed += float(mb_aux)

    assert abs(summed - full_value) < 1e-4, f"microbatch sum {summed} != single-batch {full_value}"


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_microbatch_cadence(device: torch.device):
    """
    M microbatches followed by one ``post_batch`` must produce the same EMA buffers
    as a single forward over the concatenated batch followed by ``post_batch``.
    Guarantees the per-step EMA decay is unambiguous and microbatch-independent.
    """
    torch.manual_seed(0)
    full = torch.randn((8, 4, 32), device=device)

    def _run(num_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(123)
        router = MoELinearRouter(
            d_model=32,
            num_experts=4,
            top_k=2,
            gating_function=MoERouterGatingFunction.softmax,
            ema_zscore_normalize=True,
            ema_zscore_alpha=0.7,
        ).to(device)
        router.train()
        for chunk in full.chunk(num_chunks, dim=0):
            router(chunk)
        router.post_batch()
        assert router._ema_mean is not None and router._ema_var is not None
        return router._ema_mean.clone(), router._ema_var.clone()

    mean_full, var_full = _run(1)
    mean_mb, var_mb = _run(4)
    torch.testing.assert_close(mean_full, mean_mb)
    torch.testing.assert_close(var_full, var_mb)


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_dry_run(device: torch.device):
    """``post_batch(dry_run=True)`` must leave EMA buffers unchanged AND clear accumulators."""
    router = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
    ).to(device)
    router.train()

    # First a real step so buffers move away from (0, 0).
    torch.manual_seed(0)
    router(torch.randn((4, 8, 32), device=device))
    router.post_batch()
    assert router._ema_mean is not None and router._ema_var is not None
    assert router._ema_step_count is not None
    snapshot_mean = router._ema_mean.clone()
    snapshot_sq = router._ema_var.clone()
    snapshot_step = int(router._ema_step_count.item())
    assert snapshot_step == 1

    # Now a dry-run step: forward fills the accumulator, but post_batch must not commit
    # — neither the EMA buffers NOR the step counter advance.
    router(torch.randn((4, 8, 32), device=device))
    assert router._ema_logit_sum_accum is not None
    router.post_batch(dry_run=True)
    torch.testing.assert_close(router._ema_mean, snapshot_mean)
    torch.testing.assert_close(router._ema_var, snapshot_sq)
    assert int(router._ema_step_count.item()) == snapshot_step
    # Accumulators must be reset so they don't leak into the next real step.
    assert router._ema_logit_sum_accum is None
    assert router._ema_logit_sq_sum_accum is None
    assert router._ema_token_count_accum == 0


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_state_dict_roundtrip(device: torch.device):
    """EMA buffers must round-trip through state_dict so checkpoint/resume preserves them."""
    router1 = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.5,
    ).to(device)
    router1.train()

    torch.manual_seed(0)
    for _ in range(5):
        router1(torch.randn((4, 8, 32), device=device))
        router1.post_batch()

    state_dict = router1.state_dict()
    assert "_ema_mean" in state_dict
    assert "_ema_var" in state_dict
    assert "_ema_step_count" in state_dict
    # EMA must be saved in fp32 so resume doesn't lose precision under bf16 training.
    assert state_dict["_ema_mean"].dtype == torch.float32
    assert state_dict["_ema_var"].dtype == torch.float32
    # Step counter persists so bias correction `1 / (1 - α^t)` is exact post-resume.
    assert state_dict["_ema_step_count"].dtype == torch.long
    assert int(state_dict["_ema_step_count"].item()) == 5

    router2 = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.5,
    ).to(device)
    router2.load_state_dict(state_dict)
    assert router2._ema_mean is not None and router2._ema_var is not None
    assert router2._ema_step_count is not None
    torch.testing.assert_close(router2._ema_mean, router1._ema_mean)
    torch.testing.assert_close(router2._ema_var, router1._ema_var)
    torch.testing.assert_close(router2._ema_step_count, router1._ema_step_count)
    # The Python shadow of the step count must be re-synced from the checkpointed
    # buffer on load — otherwise `_apply_ema_zscore` would apply the wrong bias
    # correction factor until the next post_batch silently re-aligned them.
    assert router2._ema_step_py == router1._ema_step_py == 5


@pytest.mark.parametrize("device", DEVICES)
def test_router_ema_bias_correction(device: torch.device):
    """
    After exactly one optimizer step, Adam-style bias correction must yield the EXACT
    batch mean / batch variance (the ``1 - α`` shrinkage cancels). This is the property
    that makes EMA effective from step 1 instead of ramping up over ~1/(1-α) steps.
    """
    torch.manual_seed(0)
    router = MoELinearRouter(
        d_model=32,
        num_experts=4,
        top_k=2,
        gating_function=MoERouterGatingFunction.softmax,
        ema_zscore_normalize=True,
        ema_zscore_alpha=0.99,  # paper-realistic value
    ).to(device)
    router.train()

    x = torch.randn((4, 16, 32), device=device)
    # Capture the actual logits the router computes (used to populate the accumulator).
    with torch.no_grad():
        logits = router.get_expert_logits(x).float().view(-1, 4)
        true_mean = logits.mean(dim=0)
        true_var = logits.var(dim=0, unbiased=False)

    router(x)
    router.post_batch()

    # Bias-corrected estimates should reproduce the exact batch statistics.
    assert router._ema_step_count is not None
    step = int(router._ema_step_count.item())
    assert step == 1
    bias_correction = 1.0 - (router.ema_zscore_alpha**step)
    mean_hat = router._ema_mean / bias_correction
    var_hat = router._ema_var / bias_correction
    torch.testing.assert_close(mean_hat, true_mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(var_hat, true_var, atol=1e-5, rtol=1e-4)

    # Without bias correction, the raw EMA buffer would be (1-α)=0.01× the true value
    # — confirm we'd be off by ~100× without the / (1 - α^t) factor.
    assert router._ema_mean.abs().max().item() < true_mean.abs().max().item() * 0.1
