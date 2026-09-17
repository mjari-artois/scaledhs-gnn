import torch

from ai4co_gnn.large_scale.partition import (
    angular_partition,
    gather_local_coordinates,
)
from ai4co_gnn.large_scale.partition import propose_adjacent_group_swaps


def test_angular_partition():

    locs = torch.rand(1, 501, 2)  # 1 batch, 501 nodes (1 depot + 500 customers), 2D coordinates
    group_size = 50
    local_indices = angular_partition(locs, group_size)
    local_locs = gather_local_coordinates(locs, local_indices)

    assert local_indices.shape == (1, 10, 51)  # 10 groups of 50 customers + depot
    assert local_locs.shape == (1, 10, 51, 2)
    print(local_indices.shape)
    print(local_locs.shape)



def test_propose_adjacent_group_swaps_preserves_partition():
      # Two groups, each containing the depot and three customers.
    groups = torch.tensor(
          [[[0, 1, 2, 3],
            [0, 4, 5, 6],
            [0, 7, 8, 9]]]
    )

    candidates = propose_adjacent_group_swaps(groups)

    # K groups produce K - 1 adjacent-swap candidates.
    assert len(candidates) == 2

    original_customers = torch.sort(groups[:, :, 1:].reshape(-1))[0]

    for candidate in candidates:
        assert candidate.shape == groups.shape
        assert torch.all(candidate[:, :, 0] == 0)

        candidate_customers = torch.sort(candidate[:, :, 1:].reshape(-1))[0]
        assert torch.equal(candidate_customers, original_customers)

      # First candidate swaps customer 3 with customer 4.
    assert torch.equal(
        candidates[0],
        torch.tensor([[[0, 1, 2, 4],[0, 3, 5, 6],[0, 7, 8, 9]]]),
      )