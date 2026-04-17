import pytest
import torch

from olmo_core.nn.moe.router import MoELinearRouter, MoERouterGatingFunction
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

    # EMA state is lazily initialized in forward.
    assert router._ema_mean is None
    assert router._ema_sq is None

    x = torch.randn((2, 4, 128), device=device)
    weights, indices, _, _ = router(x)
    assert weights.shape == (2, 4, 2)
    assert indices.shape == (2, 4, 2)

    # After one training forward, EMA state is initialized and on the right device.
    assert router._ema_mean is not None
    assert router._ema_sq is not None
    assert router._ema_mean.device.type == device.type
    assert router._ema_sq.device.type == device.type

    # Stats should have moved away from the (0, 1) init.
    assert router._ema_mean.abs().sum().item() > 0
    snapshot_mean = router._ema_mean.clone()
    snapshot_sq = router._ema_sq.clone()

    # In eval mode the EMA should NOT update.
    router.eval()
    router(torch.randn((2, 4, 128), device=device))
    torch.testing.assert_close(router._ema_mean, snapshot_mean)
    torch.testing.assert_close(router._ema_sq, snapshot_sq)


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

    # E[X^2] - (E[X])^2 should be a non-degenerate variance estimate.
    var = router._ema_sq - router._ema_mean.pow(2)
    assert (var > 0).all(), f"variance must be positive, got {var}"


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
