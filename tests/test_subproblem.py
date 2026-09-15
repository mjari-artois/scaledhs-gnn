import torch
from tensordict import TensorDict

from ai4co_gnn.large_scale.partition import angular_partition
from ai4co_gnn.large_scale.subproblem import build_local_tensordict


def test_build_local_tensordict():
    batch_size = 2
    num_customers = 500
    group_size = 50

    # Node 0 is the depot; nodes 1 through 500 are customers.
    locs = torch.rand(batch_size, num_customers + 1, 2)
    demand = torch.rand(batch_size, num_customers + 1)
    vehicle_capacity = torch.ones(batch_size, 1)

    global_td = TensorDict(
        {
            "locs": locs,
            "demand": demand,
            "vehicle_capacity": vehicle_capacity,
        },
        batch_size=[batch_size],
    )

    local_indices = angular_partition(locs, group_size=group_size)
    local_td = build_local_tensordict(global_td, local_indices)

    # Two global instances x ten groups = twenty local problems.
    assert local_td.batch_size == torch.Size([20])
    assert local_td["locs"].shape == (20, 51, 2)
    assert local_td["demand"].shape == (20, 51)
    assert local_td["vehicle_capacity"].shape == (20, 1)

    # Every local problem starts with the original depot coordinates.
    expected_depots = locs[:, :1].repeat_interleave(10, dim=0).squeeze(1)
    assert torch.equal(local_td["locs"][:, 0], expected_depots)
