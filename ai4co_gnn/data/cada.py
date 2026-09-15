"""Reader and normalizer for CADA synthetic MTVRP datasets."""

from __future__ import annotations

from pathlib import Path

import torch

from rl4co.data.utils import load_npz_to_tensordict
from tensordict import TensorDict


REQUIRED_KEYS = {
    "locs",
    "demand_backhaul",
    "demand_linehaul",
    "vehicle_capacity",
}


def transform_cada_tensordict(td: TensorDict) -> TensorDict:
    """Convert CADA fields to the canonical fields used by this project."""
    missing = REQUIRED_KEYS.difference(td.keys())
    if missing:
        raise ValueError(f"CADA dataset is missing required fields: {sorted(missing)}")

    # CADA stores the HGS reference objective as opt_cost.  The project uses
    # costs_bks when computing the optional gap-to-reference metric.
    if "opt_cost" in td.keys() and "costs_bks" not in td.keys():
        td.set("costs_bks", td["opt_cost"].clone())

    batch_size = td.batch_size
    n_nodes = td["locs"].shape[-2]

    if "open_route" not in td.keys():
        td.set("open_route", torch.zeros((*batch_size, 1), dtype=torch.bool))
    if "distance_limit" not in td.keys():
        td.set("distance_limit", torch.full((*batch_size, 1), float("inf")))
    if "time_windows" not in td.keys():
        tw = torch.zeros((*batch_size, n_nodes, 2), dtype=td["locs"].dtype)
        tw[..., 1] = float("inf")
        td.set("time_windows", tw)
    if "service_time" not in td.keys():
        td.set("service_time", torch.zeros_like(td["demand_linehaul"]))
    if "speed" not in td.keys():
        td.set("speed", torch.ones((*batch_size, 1), dtype=td["locs"].dtype))
    if "capacity_original" not in td.keys():
        td.set("capacity_original", td["vehicle_capacity"].clone())

    # Older CADA exports may omit the depot demand (node 0).
    if td["demand_linehaul"].shape[-1] == n_nodes - 1:
        zero = torch.zeros_like(td["demand_linehaul"][..., :1])
        td.set("demand_linehaul", torch.cat((zero, td["demand_linehaul"]), dim=-1))
        td.set(
            "demand_backhaul",
            torch.cat((zero, td["demand_backhaul"]), dim=-1),
        )

    return td


def load_cada_npz(path: str | Path) -> TensorDict:
    """Read one CADA `.npz` file and return the project canonical TensorDict."""
    td = load_npz_to_tensordict(str(path))
    return transform_cada_tensordict(td)
