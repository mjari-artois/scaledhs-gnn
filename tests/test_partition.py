import torch

from ai4co_gnn.large_scale.partition import (
    angular_partition,
    gather_local_coordinates,
)

def test_angular_partition():

    locs = torch.rand(1, 501, 2)  # 1 batch, 501 nodes (1 depot + 500 customers), 2D coordinates
    group_size = 50
    local_indices = angular_partition(locs, group_size)
    local_locs = gather_local_coordinates(locs, local_indices)

    assert local_indices.shape == (1, 10, 51)  # 10 groups of 50 customers + depot
    assert local_locs.shape == (1, 10, 51, 2)
    print(local_indices.shape)
    print(local_locs.shape)