from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


def load_uniform_npz(path: str | Path) -> TensorDict:
    """Load ``uniform_N500/problems_test.npz`` as an RL4CO-style TensorDict."""
    raw = np.load(path, allow_pickle=False)

    required = {"nodes", "demands", "capacities"}
    missing = required.difference(raw.files)
    if missing:
        raise KeyError(
            f"Dataset {path} is missing fields: {sorted(missing)}"
        )

    nodes = torch.as_tensor(raw["nodes"], dtype=torch.float32)
    demands = torch.as_tensor(raw["demands"], dtype=torch.float32)
    capacities = torch.as_tensor(raw["capacities"], dtype=torch.float32)

    if nodes.ndim != 3 or nodes.shape[-1] != 2:
        raise ValueError("nodes must have shape [batch_size, num_nodes, 2]")

    if demands.shape[:2] != nodes.shape[:2]:
        raise ValueError("demands must have shape [batch_size, num_nodes]")

    if capacities.shape != (nodes.shape[0],):
        raise ValueError("capacities must have shape [batch_size]")

    if torch.any(capacities <= 0):
        raise ValueError("All vehicle capacities must be positive")

    # Normalize each instance independently to match the model's capacity=1
    demand_linehaul = demands / capacities[:, None]
    demand_backhaul = torch.zeros_like(demand_linehaul)

    batch_size, num_nodes, _ = nodes.shape
    time_windows = torch.zeros(batch_size, num_nodes, 2)
    time_windows[..., 1] = float("inf")

    return TensorDict(
        {
            "locs": nodes,
            "demand_linehaul": demand_linehaul,
            "demand_backhaul": demand_backhaul,
            "vehicle_capacity": torch.ones(batch_size, 1),
            "capacity_original": capacities[:, None],
            "service_time": torch.zeros_like(demand_linehaul),
            "time_windows": time_windows,
            "distance_limit": torch.full(
                (batch_size, 1),
                float("inf"),
            ),
            "open_route": torch.zeros(batch_size, 1, dtype=torch.bool),
            "speed": torch.ones(batch_size, 1),
        },
        batch_size=[batch_size],
    )
