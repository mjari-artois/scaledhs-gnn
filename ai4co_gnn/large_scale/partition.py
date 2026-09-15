from __future__ import annotations

import math

import torch
from torch import Tensor


def angular_partition(
        locs: Tensor,
        group_size: int = 50,
        angle_offset: float = 0.0,
) -> Tensor:
    """  Split customers into spatial groups using their angle from the depot.  """
    batch_size, num_nodes, _ = locs.shape
    num_customers = num_nodes - 1  # Exclude depot
    num_groups = num_customers // group_size

    depot = locs[:, :1] #[B, 1, 2]
    customers= locs[:, 1:] #[B, N, 2]

    relative_position = customers - depot
    angles = torch.atan2(
        relative_position[..., 1],
        relative_position[..., 0]
    )

    if angle_offset != 0.0:
        angles = torch.remainder(
            angles - angle_offset,
            2.0 * math.pi,
        )

    sorted_customer_positions = torch.argsort(angles, dim=-1) #[B, N]
    sorted_customer_indices = sorted_customer_positions + 1  # Shift by 1 to account for depot

    customer_groups = sorted_customer_indices.reshape(
        batch_size,
        num_groups,
        group_size,
    )
    depot_indices = torch.zeros(
        batch_size,
        num_groups,
        1,
        dtype=torch.long,
        device=locs.device,
    )
    local_indices = torch.cat(
        [depot_indices, customer_groups],
        dim=-1
    )

    return local_indices

def gather_local_coordinates(
        locs: Tensor,
        local_indices: Tensor
) -> Tensor:
    """ Gather local coordinates based on local indices. """
    batch_size, num_groups, local_size = local_indices.shape

    expand_locs = locs.unsqueeze(1).expand(
        batch_size,
        num_groups,
        locs.size(1),
        locs.size(2),
    )
    expand_indices = local_indices.unsqueeze(-1).expand(
        batch_size,
        num_groups,
        local_size,
        locs.size(-1),
    )

    local_locs = torch.gather(
        expand_locs,
        dim=2,
        index=expand_indices
    )

    return local_locs


