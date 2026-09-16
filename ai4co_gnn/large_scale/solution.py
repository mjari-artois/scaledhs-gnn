"""Reconstruct and validate global CVRP solutions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from torch import Tensor


@dataclass
class SolutionMetrics:
    """Validation and cost information for one CVRP solution."""
    cost: float
    feasible: bool
    served_customers: int
    expected_customers: int
    duplicate_customers: list[int]
    missing_customers: list[int]
    capacity_violations: list[float]
    invalid_routes: list[list[int]]


def _split_action_sequence(
    actions: Iterable[int],
    depot: int = 0,
) -> list[list[int]]:
    """Split a depot-delimited action sequence into depot-to-depot routes."""
    sequence = [int(node) for node in actions]
    routes: list[list[int]] = []
    current_route: list[int] = []

    for node in sequence:
        if node == depot:
            if current_route:
                current_route.append(depot)
                routes.append(current_route)
                current_route = []
            current_route = [depot]
        elif current_route:
            current_route.append(node)
        else:
            # Be tolerant if a solver omits the initial depot.
            current_route = [depot, node]

    if current_route and len(current_route) > 1:
        if current_route[-1] != depot:
            current_route.append(depot)
        routes.append(current_route)

    return routes


def actions_to_routes(
    global_actions: Tensor,
    num_instances: int,
    num_groups: int,
    depot: int = 0,
) -> list[list[list[int]]]:
    """Convert flattened local actions into routes for each global instance"""
    if global_actions.ndim != 2:
        raise ValueError(
            "Route reconstruction expects actions with shape [B*G, T]. "
            "Select a POMO start before calling this function."
        )

    expected_batch = num_instances * num_groups
    if global_actions.size(0) != expected_batch:
        raise ValueError(
            f"Expected {expected_batch} local action sequences, "
            f"received {global_actions.size(0)}"
        )

    actions_cpu = global_actions.detach().cpu()
    solutions: list[list[list[int]]] = []

    for instance_idx in range(num_instances):
        instance_routes: list[list[int]] = []
        start = instance_idx * num_groups
        end = start + num_groups

        for action_sequence in actions_cpu[start:end]:
            instance_routes.extend(
                _split_action_sequence(action_sequence.tolist(), depot=depot)
            )

        solutions.append(instance_routes)

    return solutions


def evaluate_solution(
    routes: list[list[int]],
    locs: Tensor,
    demand: Tensor,
    vehicle_capacity: Tensor | float,
    expected_customers: int | None = None,
    max_vehicles: int | None = None,
    depot: int = 0,
) -> SolutionMetrics:
    """Validate one CVRP solution and calculate its Euclidean cost."""
    locs_cpu = locs.detach().cpu()
    demand_cpu = demand.detach().cpu().flatten()

    if isinstance(vehicle_capacity, Tensor):
        capacity = float(vehicle_capacity.detach().cpu().flatten()[0])
    else:
        capacity = float(vehicle_capacity)

    if expected_customers is None:
        expected_customers = locs_cpu.shape[0] - 1

    visited: list[int] = []
    invalid_routes: list[list[int]] = []
    capacity_violations: list[float] = []
    total_cost = 0.0

    for route in routes:
        route = [int(node) for node in route]

        valid_route = (
            len(route) >= 2
            and route[0] == depot
            and route[-1] == depot
            and all(0 <= node < locs_cpu.shape[0] for node in route)
        )

        if not valid_route:
            invalid_routes.append(route)
            continue

        route_customers = [node for node in route if node != depot]
        visited.extend(route_customers)

        route_demand = float(demand_cpu[route_customers].sum())
        if route_demand > capacity + 1e-6:
            capacity_violations.append(route_demand - capacity)

        for start, end in zip(route[:-1], route[1:]):
            start_xy = locs_cpu[start]
            end_xy = locs_cpu[end]
            total_cost += math.hypot(
                float(start_xy[0] - end_xy[0]),
                float(start_xy[1] - end_xy[1]),
            )

    counts: dict[int, int] = {}
    for customer in visited:
        counts[customer] = counts.get(customer, 0) + 1

    duplicate_customers = sorted(
        customer for customer, count in counts.items() if count > 1
    )
    missing_customers = sorted(
        customer
        for customer in range(1, expected_customers + 1)
        if counts.get(customer, 0) == 0
    )

    too_many_vehicles = max_vehicles is not None and len(routes) > max_vehicles
    feasible = not (
        invalid_routes
        or capacity_violations
        or duplicate_customers
        or missing_customers
        or too_many_vehicles
    )

    return SolutionMetrics(
        cost=total_cost,
        feasible=feasible,
        served_customers=len(visited),
        expected_customers=expected_customers,
        duplicate_customers=duplicate_customers,
        missing_customers=missing_customers,
        capacity_violations=capacity_violations,
        invalid_routes=invalid_routes,
    )
