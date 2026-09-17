import torch
from tensordict import TensorDict
from torch import nn

from ai4co_gnn.large_scale.coordinator import AngularSubproblemCoordinator


class DummyEnvironment:
    """Minimal environment matching the solver interface."""

    def reset(self, td):
        return td


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, td, env, phase, num_starts):
        self.call_count += 1

        batch_size = td.batch_size[0]
        device = td.device

        # Local node IDs: depot -> customer 1 -> customer 2 -> depot.
        actions = torch.tensor(
            [[0, 1, 2, 0]],
            dtype=torch.long,
            device=device,
        ).expand(batch_size, -1)

        return {
            "actions": actions,
            "reward": torch.zeros(batch_size, device=device),
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # NeuralLocalSolver calls model.policy(...), like the real POMO model.
        self.policy = DummyPolicy()


def test_angular_subproblem_coordinator():
    batch_size = 2
    num_customers = 500
    group_size = 50

    # Node 0 is the depot; nodes 1 through 500 are customers.
    global_td = TensorDict(
        {
            "locs": torch.rand(batch_size, num_customers + 1, 2),
            "demand": torch.rand(batch_size, num_customers + 1),
            "vehicle_capacity": torch.ones(batch_size, 1),
        },
        batch_size=[batch_size],
    )

    model = DummyModel()
    env = DummyEnvironment()

    coordinator = AngularSubproblemCoordinator(
        model=model,
        env=env,
        group_size=group_size,
        num_starts=1,
        device="cpu",
        demand_key="demand",
    )

    result = coordinator.solve(global_td)

    # Two global instances and ten local groups per instance.
    assert result.local_indices.shape == (batch_size, 10, 51)
    assert result.local_td.batch_size == torch.Size([20])
    assert result.global_actions.shape == (20, 4)

    # The policy was called once for all 20 local problems.
    # One initial solve plus one solve for each of the nine adjacent swaps.
    assert model.policy.call_count == 10


def test_coordinator_keeps_baseline_when_swap_is_tied():
    batch_size = 1
    num_customers = 4
    group_size = 2

    global_td = TensorDict(
        {
            "locs": torch.zeros(batch_size, num_customers + 1, 2),
            "demand": torch.zeros(batch_size, num_customers + 1),
            "vehicle_capacity": torch.ones(batch_size, 1),
        },
        batch_size=[batch_size],
    )

    coordinator = AngularSubproblemCoordinator(
        model=DummyModel(),
        env=DummyEnvironment(),
        group_size=group_size,
        num_starts=1,
        device="cpu",
        demand_key="demand",
        max_vehicles=2,
    )

    result = coordinator.solve(global_td)

    expected_initial_partition = torch.tensor(
        [[[0, 1, 2], [0, 3, 4]]]
    )
    assert torch.equal(result.local_indices, expected_initial_partition)
    assert result.global_actions.shape == (2, 4)


def test_coordinator_refinement_controls_candidate_count():
    global_td = TensorDict(
        {
            "locs": torch.zeros(1, 5, 2),
            "demand": torch.zeros(1, 5),
            "vehicle_capacity": torch.ones(1, 1),
        },
        batch_size=[1],
    )

    baseline_model = DummyModel()
    baseline = AngularSubproblemCoordinator(
        model=baseline_model,
        env=DummyEnvironment(),
        group_size=2,
        num_starts=1,
        device="cpu",
        demand_key="demand",
        refinement_enabled=False,
    )
    baseline.solve(global_td)
    assert baseline_model.policy.call_count == 1

    limited_model = DummyModel()
    limited = AngularSubproblemCoordinator(
        model=limited_model,
        env=DummyEnvironment(),
        group_size=2,
        num_starts=1,
        device="cpu",
        demand_key="demand",
        refinement_enabled=True,
        max_candidates=1,
    )
    limited.solve(global_td)
    assert limited_model.policy.call_count == 2
