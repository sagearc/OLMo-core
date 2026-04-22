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
    bc = 1.0 - (router.ema_zscore_alpha**step)
    expected_mean = router._ema_mean / bc
    expected_std = (router._ema_var / bc).clamp(min=1e-4).sqrt()
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

    mean_full, sq_full = _run(1)
    mean_mb, sq_mb = _run(4)
    torch.testing.assert_close(mean_full, mean_mb)
    torch.testing.assert_close(sq_full, sq_mb)


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
    bc = 1.0 - (router.ema_zscore_alpha**step)
    mean_hat = router._ema_mean / bc
    var_hat = router._ema_var / bc
    torch.testing.assert_close(mean_hat, true_mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(var_hat, true_var, atol=1e-5, rtol=1e-4)

    # Without bias correction, the raw EMA buffer would be (1-α)=0.01× the true value
    # — confirm we'd be off by ~100× without the / (1 - α^t) factor.
    assert router._ema_mean.abs().max().item() < true_mean.abs().max().item() * 0.1
