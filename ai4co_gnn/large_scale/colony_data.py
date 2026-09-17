"""Small deterministic CVRP episode generation for colony training and tests."""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from ai4co_gnn.large_scale.train_colony import ColonyEpisode
from ai4co_gnn.large_scale.solution import evaluate_solution


def _pack_routes(order: Sequence[int], demands: torch.Tensor, capacity: float) -> list[list[int]]:
    routes: list[list[int]] = []
    current = [0]
    load = 0.0
    for customer in order:
        demand = float(demands[customer])
        if current != [0] and load + demand > capacity + 1e-8:
            routes.append(current + [0])
            current = [0]
            load = 0.0
        current.append(int(customer))
        load += demand
    if len(current) > 1:
        routes.append(current + [0])
    return routes


def make_synthetic_episodes(
    count: int,
    num_customers: int = 20,
    *,
    capacity: float = 1.0,
    seed: int = 0,
) -> list[ColonyEpisode]:
    """Generate feasible uniform Euclidean CVRP episodes.

    The reference is an angularly ordered feasible solution.  This is intended
    for local training and unit tests; replace it with held-out reference
    solutions for research measurements.
    """
    if count < 1 or num_customers < 1:
        raise ValueError("count and num_customers must be positive")
    generator = torch.Generator().manual_seed(seed)
    episodes: list[ColonyEpisode] = []

    for episode_index in range(count):
        locs = torch.rand((num_customers + 1, 2), generator=generator)
        demands = torch.rand((num_customers + 1,), generator=generator) * (capacity / 4.0)
        demands[0] = 0.0
        angles = torch.atan2(locs[1:, 1] - locs[0, 1], locs[1:, 0] - locs[0, 0])
        angular_order = (torch.argsort(angles) + 1).tolist()
        reference_routes = _pack_routes(angular_order, demands, capacity)

        shuffled = angular_order[:]
        random.Random(seed + episode_index).shuffle(shuffled)
        initial_routes = _pack_routes(shuffled, demands, capacity)

        global_td = TensorDict(
            {
                "locs": locs.unsqueeze(0),
                "demand": demands.unsqueeze(0),
                "vehicle_capacity": torch.tensor([[capacity]], dtype=torch.float32),
            },
            batch_size=[1],
        )
        reference_cost = evaluate_solution(
            reference_routes,
            locs,
            demands,
            capacity,
            expected_customers=num_customers,
        ).cost
        episodes.append(
            ColonyEpisode(
                global_td=global_td,
                initial_routes=initial_routes,
                reference_cost=reference_cost,
            )
        )
    return episodes


def episodes_from_tensordict(
    data: TensorDict,
    *,
    max_instances: int | None = None,
    seed: int = 0,
) -> list[ColonyEpisode]:
    """Convert loaded uniform CVRP data into independent colony episodes."""
    count = data.batch_size[0]
    if max_instances is not None:
        count = min(count, max_instances)
    episodes: list[ColonyEpisode] = []
    for index in range(count):
        episodes.append(_episode_from_instance(data[index : index + 1], index, seed))
    return episodes


def _episode_from_instance(
    instance: TensorDict,
    index: int,
    seed: int,
) -> ColonyEpisode:
        locs = instance["locs"][0]
        if "demand" in instance:
            demands = instance["demand"][0]
        else:
            demands = instance["demand_linehaul"][0]
        capacity = float(instance["vehicle_capacity"].reshape(-1)[0].item())
        customers = list(range(1, locs.shape[0]))
        angles = torch.atan2(
            locs[1:, 1] - locs[0, 1], locs[1:, 0] - locs[0, 0]
        )
        ordered = (torch.argsort(angles) + 1).tolist()
        reference_routes = _pack_routes(ordered, demands, capacity)
        shuffled = ordered[:]
        random.Random(seed + index).shuffle(shuffled)
        initial_routes = _pack_routes(shuffled, demands, capacity)
        reference_cost = evaluate_solution(
            reference_routes,
            locs,
            demands,
            capacity,
            expected_customers=len(customers),
        ).cost
        return ColonyEpisode(
            global_td=instance,
            initial_routes=initial_routes,
            reference_cost=reference_cost,
        )


class TensorDictEpisodeDataset(Dataset[ColonyEpisode]):
    """Lazy episode view over an in-memory TensorDict.

    Generation stays in RAM and no intermediate per-instance files or list of
    100K TensorDict objects is created.  Only the requested batch is wrapped as
    colony episodes by ``__getitem__``.
    """

    def __init__(self, data: TensorDict, *, max_instances: int | None = None, seed: int = 0) -> None:
        self.data = data
        self.length = data.batch_size[0] if max_instances is None else min(int(max_instances), data.batch_size[0])
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> ColonyEpisode:
        return _episode_from_instance(self.data[index : index + 1], index, self.seed)


def sequential_repair(local_td: TensorDict) -> dict[str, torch.Tensor]:
    """Deterministic repair fallback for smoke tests and config validation.

    Production training should pass a function wrapping the frozen neural local
    solver instead.
    """
    local_size = int(local_td["locs"].shape[1])
    actions = torch.arange(local_size, device=local_td.device).unsqueeze(0)
    actions = torch.cat((actions, torch.zeros((1, 1), dtype=torch.long, device=actions.device)), dim=1)
    return {"actions": actions}
