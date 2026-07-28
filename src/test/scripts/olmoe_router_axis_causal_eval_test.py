import torch
import torch.nn.functional as F

from olmo_core.nn.moe.mlp import DroplessMoEMLP
from scripts.olmoe_router_axis_causal_eval import (
    GroupedProjectionHook,
    InterventionCondition,
    apply_grouped_projection,
    implicit_orthogonal_eigendirections,
    quadratic_energy,
    repair_olmoe_gate_up_layout,
    torch_inference_gather,
    torch_inference_scatter,
)


def test_grouped_projection_is_expert_specific_and_exact():
    x = torch.tensor(
        [
            [3.0, 4.0, 1.0],
            [2.0, -1.0, 5.0],
            [7.0, 2.0, -3.0],
        ]
    )
    counts = torch.tensor([2, 1])
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    identity, _ = apply_grouped_projection(x, counts, directions, alpha=0.0)
    projected, diagnostics = apply_grouped_projection(x, counts, directions, alpha=1.0)

    assert identity.data_ptr() == x.data_ptr()
    torch.testing.assert_close(
        projected,
        torch.tensor([[0.0, 4.0, 1.0], [0.0, -1.0, 5.0], [7.0, 0.0, -3.0]]),
    )
    assert diagnostics["projection_residual_max"] == 0.0


def test_projection_hook_matches_manual_expert_computation():
    torch.manual_seed(4)
    mlp = DroplessMoEMLP(d_model=4, hidden_size=3, num_experts=2)
    x = torch.randn(3, 4)
    counts = torch.tensor([2, 1])
    directions = F.normalize(torch.randn(2, 4), dim=-1)
    condition = InterventionCondition("oracle", "router", 1.0)

    projected, _ = apply_grouped_projection(x, counts, directions, alpha=1.0)
    w1 = mlp.w1.view(2, 3, 4)
    w2 = mlp.w2.view(2, 3, 4)
    w3 = mlp.w3.view(2, 3, 4)
    expected_parts = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        expert_x = projected[start : start + count]
        hidden = F.silu(expert_x @ w1[expert].t()) * (expert_x @ w3[expert].t())
        expected_parts.append(hidden @ w2[expert])
        start += count
    expected = torch.cat(expected_parts)

    hook = GroupedProjectionHook(mlp, directions, condition, validate_manual=True)
    with hook:
        actual = mlp(x, counts)

    torch.testing.assert_close(actual, expected)
    assert hook.diagnostics()["manual_output_max_abs_error"] < 1e-6
    restored = mlp(x, counts)
    assert not torch.equal(restored, actual)


def test_implicit_control_recovers_known_orthogonal_energy_axes():
    # Router is e0.  The two strongest legal directions are e2 and e1.
    router = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    w1 = torch.zeros(1, 4, 4)
    w3 = torch.zeros(1, 4, 4)
    w1[0] = torch.diag(torch.tensor([9.0, 2.0, 5.0, 1.0]))

    directions, eigenvalues = implicit_orthogonal_eigendirections(
        w1,
        w3,
        router,
        candidate_count=2,
        iterations=24,
        seed=7,
    )

    assert directions.shape == (1, 2, 4)
    assert float((directions @ router.unsqueeze(-1)).abs().max()) < 1e-5
    torch.testing.assert_close(eigenvalues[0], torch.tensor([25.0, 4.0]), rtol=1e-3, atol=1e-3)
    energies = quadratic_energy(directions[:, 0], w1, w3)
    torch.testing.assert_close(energies, torch.tensor([25.0]), rtol=1e-3, atol=1e-3)


def test_hf_gate_up_layout_repair_is_per_expert_transpose():
    # E=2, D=3, H=2.  HF conversion supplies [E*D, H]; native expects [E*H, D].
    key = "blocks.0.feed_forward_moe.experts.mlp.w1"
    expert_major = torch.arange(12).view(2, 3, 2)
    converted = {key: expert_major.view(6, 2)}
    native = {key: torch.empty(4, 3)}

    repaired = repair_olmoe_gate_up_layout(converted, native)

    assert repaired == [key]
    torch.testing.assert_close(converted[key].view(2, 2, 3), expert_major.transpose(1, 2))


def test_pure_torch_routing_permutation_matches_weighted_topk_sum():
    # Three tokens, top-2 expert assignments. `indices` is the expert-sorted
    # permutation of the six flattened assignments.
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    expert_ids = torch.tensor([1, 0, 2, 1, 0, 2])
    weights = torch.tensor([0.7, 0.3, 0.4, 0.6, 0.2, 0.8])
    bin_ids, indices = torch.sort(expert_ids)
    bins = torch.bincount(expert_ids, minlength=3).cumsum(0)

    grouped = torch_inference_gather(x, indices, bin_ids, bins, top_k=2)
    restored = torch_inference_scatter(grouped, indices, bin_ids, weights, bins, top_k=2)

    expected = x * weights.view(3, 2).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(restored, expected)
