from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Sequence

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from ai4co_gnn.large_scale.solution import evaluate_solution
from ai4co_gnn.large_scale.subproblem import build_local_tensordict


@dataclass(frozen=True)
class ColonyCandidate:
    route_ids: tuple[int, ...]
    customers: tuple[int, ...]
    features: Tensor
    pheromone: float


class ColonyCVRP:
    """Run colony-guided repairs around a frozen small-instance policy."""

    def __init__(
        self,
        global_td: TensorDictBase,
        initial_routes: Sequence[Sequence[int]],
        repair_fn: Callable[[TensorDictBase], Tensor | dict],
        selector: nn.Module | None = None,
        *,
        max_local_customers: int = 50,
        candidate_pool_size: int = 32,
        n_ants: int = 4,
        rounds: int = 20,
        evaporation: float = 0.05,
        pheromone_alpha: float = 1.0,
        pheromone_min: float = 1.0,
        pheromone_max: float = 10.0,
        exploration: float = 0.10,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:

        self.global_td = global_td
        self.device = (
            torch.device(device)
            if device is not None
            else global_td["locs"].device
        )
        self.locs = global_td["locs"][0].to(self.device)
        self.demands = self._demands(global_td).to(self.device)
        self.capacity = float(global_td["vehicle_capacity"].reshape(-1)[0].item())
        self.routes = [list(map(int, route)) for route in initial_routes]
        self.best_routes = [route[:] for route in self.routes]
        self.repair_fn = repair_fn
        self.selector = selector.to(self.device).eval() if selector is not None else None
        self.max_local_customers = max_local_customers
        self.candidate_pool_size = candidate_pool_size
        self.n_ants = n_ants
        self.rounds = rounds
        self.evaporation = evaporation
        self.pheromone_alpha = pheromone_alpha
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.exploration = exploration
        self.pheromone: dict[tuple[int, int], float] = {}

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self._validate(self.routes)
        self.initial_cost = self.solution_cost(self.routes)
        self.best_cost = self.initial_cost

    @staticmethod
    def _demands(global_td: TensorDictBase) -> Tensor:
        if "demand" in global_td:
            return global_td["demand"][0]
        if "demand_linehaul" in global_td:
            return global_td["demand_linehaul"][0]
        raise KeyError("global_td requires 'demand' or 'demand_linehaul'")

    @property
    def num_customers(self) -> int:
        return self.locs.shape[0] - 1

    def _validate(self, routes: Sequence[Sequence[int]]) -> None:
        metrics = evaluate_solution(
            [list(route) for route in routes], self.locs, self.demands,
            self.capacity, expected_customers=self.num_customers,
        )
        if not metrics.feasible:
            raise ValueError(
                "Infeasible CVRP routes: "
                f"missing={metrics.missing_customers}, "
                f"duplicates={metrics.duplicate_customers}, "
                f"capacity={metrics.capacity_violations}, "
                f"invalid={metrics.invalid_routes}"
            )

    def _route_customers(self, route: Sequence[int]) -> tuple[int, ...]:
        return tuple(node for node in route if node != 0)

    def _route_cost(self, route: Sequence[int]) -> float:
        return sum(
            math.hypot(
                float(self.locs[a, 0] - self.locs[b, 0]),
                float(self.locs[a, 1] - self.locs[b, 1]),
            )
            for a, b in zip(route[:-1], route[1:])
        )

    def solution_cost(self, routes: Sequence[Sequence[int]]) -> float:
        return sum(self._route_cost(route) for route in routes)

    def _load(self, route: Sequence[int]) -> float:
        customers = self._route_customers(route)
        return float(self.demands[list(customers)].sum().item()) if customers else 0.0

    def _center(self, route: Sequence[int]) -> Tensor:
        customers = self._route_customers(route)
        return self.locs[list(customers)].mean(0) if customers else self.locs[0]

    @staticmethod
    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _candidate_pheromone(self, route_ids: Sequence[int]) -> float:
        groups = [self._route_customers(self.routes[i]) for i in route_ids]
        values = [
            self.pheromone.get(self._key(a, b), self.pheromone_min)
            for i, left in enumerate(groups)
            for right in groups[i + 1:]
            for a in left for b in right
        ]
        return sum(values) / len(values) if values else self.pheromone_min

    def _features(self, route_ids: Sequence[int]) -> Tensor:
        routes = [self.routes[i] for i in route_ids]
        customers = [c for route in routes for c in self._route_customers(route)]
        points = self.locs[list(customers)] if customers else self.locs[:1]
        spread = torch.linalg.vector_norm(points.max(0).values - points.min(0).values)
        depot_distance = torch.linalg.vector_norm(points - self.locs[0], dim=-1).mean()
        return torch.tensor([
            len(customers) / self.max_local_customers,
            len(routes) / max(1, len(self.routes)),
            self.solution_cost(routes) / max(self.initial_cost, 1e-8),
            sum(self._load(r) for r in routes) / max(self.capacity * len(routes), 1e-8),
            float(spread), float(depot_distance),
            math.log(max(self._candidate_pheromone(route_ids), 1e-8)),
        ], dtype=torch.float32, device=self.device)

    def propose_candidates(self) -> list[ColonyCandidate]:
        centers = [self._center(route) for route in self.routes]
        result: list[ColonyCandidate] = []
        for rid, route in enumerate(self.routes):
            own = self._route_customers(route)
            if 0 < len(own) <= self.max_local_customers:
                ids = (rid,)
                result.append(ColonyCandidate(ids, own, self._features(ids), self._candidate_pheromone(ids)))
            nearby = sorted(
                (j for j in range(len(self.routes)) if j != rid),
                key=lambda j: float(torch.linalg.vector_norm(centers[rid] - centers[j])),
            )
            for other in nearby:
                customers = own + self._route_customers(self.routes[other])
                if len(customers) <= self.max_local_customers:
                    ids = tuple(sorted((rid, other)))
                    result.append(ColonyCandidate(ids, customers, self._features(ids), self._candidate_pheromone(ids)))
                    break
        random.shuffle(result)
        return result[:self.candidate_pool_size]

    def select_candidate(
        self,
        candidates: Sequence[ColonyCandidate],
        return_trace: bool = False,
    ) -> ColonyCandidate | tuple[ColonyCandidate, Tensor, Tensor] | None:
        if not candidates:
            return None
        features = torch.stack([c.features for c in candidates])
        scores = self.selector(features).reshape(-1) if self.selector is not None else torch.zeros(len(candidates), device=self.device)
        pheromone = torch.tensor([c.pheromone for c in candidates], device=self.device).clamp_min(1e-8)
        probabilities = torch.softmax(scores + self.pheromone_alpha * pheromone.log(), 0)
        uniform = torch.full_like(probabilities, 1.0 / len(candidates))
        probabilities = (1 - self.exploration) * probabilities + self.exploration * uniform
        distribution = torch.distributions.Categorical(probs=probabilities)
        selected = distribution.sample()
        candidate = candidates[int(selected)]
        if return_trace:
            return candidate, distribution.log_prob(selected), distribution.entropy()
        return candidate

    @staticmethod
    def _split_actions(actions: Sequence[int]) -> list[list[int]]:
        routes: list[list[int]] = []
        current: list[int] = []
        for node in map(int, actions):
            if node == 0:
                if current:
                    current.append(0)
                    routes.append(current)
                current = [0]
            elif current:
                current.append(node)
            else:
                current = [0, node]
        if len(current) > 1:
            if current[-1] != 0:
                current.append(0)
            routes.append(current)
        return routes

    @torch.inference_mode()
    def _repair(self, candidate: ColonyCandidate) -> list[list[int]]:
        customers = list(candidate.customers)
        indices = torch.tensor([[[0] + customers]], dtype=torch.long, device=self.global_td.device)
        local_td = build_local_tensordict(self.global_td, indices)
        output = self.repair_fn(local_td)
        actions = output["actions"] if isinstance(output, dict) else output
        if not isinstance(actions, Tensor):
            raise TypeError("repair_fn must return a Tensor or a dict containing 'actions'")
        actions = actions.detach().reshape(-1, actions.shape[-1])[0].to(indices.device, torch.long)
        if torch.any(actions < 0) or torch.any(actions >= indices.shape[-1]):
            raise ValueError("Frozen policy returned an invalid local node ID")
        return self._split_actions(indices[0, 0][actions].tolist())

    def _valid_local_repair(self, candidate: ColonyCandidate, routes: Sequence[Sequence[int]]) -> bool:
        expected = set(candidate.customers)
        observed = [node for route in routes for node in route if node != 0]
        return (
            all(route and route[0] == 0 and route[-1] == 0 for route in routes)
            and all(self._load(route) <= self.capacity + 1e-6 for route in routes)
            and set(observed) == expected
            and len(observed) == len(set(observed))
        )

    def _reinforce(self, candidate: ColonyCandidate, improvement: float, old_cost: float) -> None:
        credit = min(1.0, improvement / max(old_cost, 1e-8))
        for i, left in enumerate(candidate.customers):
            for right in candidate.customers[i + 1:]:
                key = self._key(left, right)
                old = self.pheromone.get(key, self.pheromone_min)
                value = (1 - self.evaporation) * old + self.evaporation * (
                    self.pheromone_min + credit * (self.pheromone_max - self.pheromone_min)
                )
                self.pheromone[key] = float(max(self.pheromone_min, min(self.pheromone_max, value)))

    def _evaporate(self) -> None:
        for key, value in self.pheromone.items():
            self.pheromone[key] = float(
                max(self.pheromone_min, (1 - self.evaporation) * value + self.evaporation * self.pheromone_min)
            )

    def _try_candidate(self, candidate: ColonyCandidate) -> bool:
        replacement = self._repair(candidate)
        if not self._valid_local_repair(candidate, replacement):
            return False
        old_selected = [self.routes[i] for i in candidate.route_ids]
        old_cost = self.solution_cost(old_selected)
        new_cost = self.solution_cost(replacement)
        improvement = old_cost - new_cost
        if improvement <= 1e-8:
            return False
        selected = set(candidate.route_ids)
        updated = [route for i, route in enumerate(self.routes) if i not in selected] + [list(r) for r in replacement]
        try:
            self._validate(updated)
        except ValueError:
            return False
        self.routes = updated
        self._reinforce(candidate, improvement, old_cost)
        cost = self.solution_cost(updated)
        if cost < self.best_cost:
            self.best_cost, self.best_routes = cost, [route[:] for route in updated]
        return True

    def run(self) -> dict:
        accepted = attempted = 0
        for _ in range(self.rounds):
            self._evaporate()
            for _ in range(self.n_ants):
                candidate = self.select_candidate(self.propose_candidates())
                if candidate is None:
                    continue
                attempted += 1
                accepted += int(self._try_candidate(candidate))
        self._validate(self.best_routes)
        return {
            "routes": self.best_routes,
            "cost": self.best_cost,
            "initial_cost": self.initial_cost,
            "accepted_repairs": accepted,
            "attempted_repairs": attempted,
            "pheromone_entries": len(self.pheromone),
        }
