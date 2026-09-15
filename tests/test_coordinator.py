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
    )

    result = coordinator.solve(global_td)

    # Two global instances and ten local groups per instance.
    assert result.local_indices.shape == (batch_size, 10, 51)
    assert result.local_td.batch_size == torch.Size([20])
    assert result.global_actions.shape == (20, 4)

    # The policy was called once for all 20 local problems.
    assert model.policy.call_count == 1
