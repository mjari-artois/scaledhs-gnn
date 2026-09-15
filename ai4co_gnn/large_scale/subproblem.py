from __future__ import annotations

from collections.abc import Iterable

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor


# Fields with one value per node and shape [batch, num_nodes, ...].
NODE_FIELDS = {
    "locs",
    "demand",
    "demand_linehaul",
    "demand_backhaul",
    "service_time",
    "time_windows",
}


def build_local_tensordict(
    global_td: TensorDictBase,
    local_indices: Tensor,
    node_fields: Iterable[str] = NODE_FIELDS,
) -> TensorDict:
    """Gather all local problems in parallel. """

    batch_size, num_groups, local_size = local_indices.shape


    node_fields = set(node_fields)
    local_indices = local_indices.to(device=global_td.device, dtype=torch.long)
    local_data: dict[str, Tensor] = {}

    for key in global_td.keys():
        value = global_td[key]

        if not isinstance(value, Tensor):
            continue

        if key in node_fields:
            if value.ndim < 2 or value.shape[0] != batch_size:
                raise ValueError(
                    f"Node field '{key}' must have shape [B, num_nodes, ...]"
                )
            grouped_value = value.unsqueeze(1).expand(
                batch_size,
                num_groups,
                *value.shape[1:],
            )

            gather_index = local_indices
            for _ in range(value.ndim - 2):
                gather_index = gather_index.unsqueeze(-1)

            gather_index = gather_index.expand(
                batch_size,
                num_groups,
                local_size,
                *value.shape[2:],
            )

            local_value = torch.gather(
                grouped_value,
                dim=2,
                index=gather_index,
            )

            local_data[key] = local_value.reshape(
                batch_size * num_groups,
                local_size,
                *value.shape[2:],
            )

        elif value.ndim >= 1 and value.shape[0] == batch_size:
            # Copy instance-level fields
            local_value = value.unsqueeze(1).expand(
                batch_size,
                num_groups,
                *value.shape[1:],
            )

            local_data[key] = local_value.reshape(
                batch_size * num_groups,
                *value.shape[1:],
            )

    return TensorDict(
        local_data,
        batch_size=[batch_size * num_groups],
        device=global_td.device,
    )
