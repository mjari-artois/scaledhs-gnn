from __future__ import annotations

import math

import numpy as np
import torch

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tensordict.tensordict import TensorDict
from torch import Tensor

from ai4co_gnn.env.pdptw.generator import PDPTWGenerator


DIST_SCALE = 10_000
TIME_SCALE = 10_000
CAP_SCALE = 10_000


def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _build_fallback_solution(instance: TensorDict) -> tuple[Tensor, Tensor]:
    pickup_idx = _as_numpy(instance["pickup_idx"]).astype(np.int64)
    delivery_idx = _as_numpy(instance["delivery_idx"]).astype(np.int64)
    locs = _as_numpy(instance["locs"]).astype(np.float64)
    if pickup_idx.ndim == 2:
        pickup_idx = pickup_idx[0]
    if delivery_idx.ndim == 2:
        delivery_idx = delivery_idx[0]
    if locs.ndim == 3:
        locs = locs[0]

    action = []
    for p, d in zip(pickup_idx.tolist(), delivery_idx.tolist()):
        action.extend([p, d, 0])

    # Approximate geometric route length for fallback sequence.
    total = 0.0
    curr = 0
    for nxt in action:
        total += float(np.linalg.norm(locs[curr] - locs[nxt]))
        curr = nxt
    if curr != 0:
        total += float(np.linalg.norm(locs[curr] - locs[0]))

    return Tensor(action).long(), torch.tensor(float(total))


def solve(
    instance: TensorDict,
    max_runtime: float = 10.0,
    first_solution_strategy: str = "PARALLEL_CHEAPEST_INSERTION",
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> tuple[Tensor, Tensor]:
    """Solve a single PDPTW instance with OR-Tools and return (actions, cost)."""
    locs = _as_numpy(instance["locs"]).astype(np.float64)
    tw = _as_numpy(instance["time_windows"]).astype(np.float64)
    service = _as_numpy(instance["service_time"]).astype(np.float64)
    if locs.ndim == 3:
        locs = locs[0]
    if tw.ndim == 3:
        tw = tw[0]
    if service.ndim == 2:
        service = service[0]
    speed = float(_as_numpy(instance["speed"]).reshape(-1)[0])

    demand_pickup = _as_numpy(instance["pickup_demand"]).astype(np.float64)
    demand_delivery = _as_numpy(instance["delivery_demand"]).astype(np.float64)
    if demand_pickup.ndim == 2:
        demand_pickup = demand_pickup[0]
    if demand_delivery.ndim == 2:
        demand_delivery = demand_delivery[0]
    signed_demand = demand_pickup - demand_delivery

    vehicle_capacity = float(_as_numpy(instance["vehicle_capacity"]).reshape(-1)[0])
    pickup_idx = _as_numpy(instance["pickup_idx"]).astype(np.int64)
    pair_index = _as_numpy(instance["pair_index"]).astype(np.int64)
    if pickup_idx.ndim == 2:
        pickup_idx = pickup_idx[0]
    if pair_index.ndim == 2:
        pair_index = pair_index[0]

    n_nodes = locs.shape[0]
    num_vehicles = max(1, n_nodes - 1)

    # Distances for objective.
    dist = np.linalg.norm(locs[:, None, :] - locs[None, :, :], axis=-1)
    dist_i = np.rint(dist * DIST_SCALE).astype(np.int64)

    # Time = travel + service at source node.
    travel = dist / max(speed, 1e-9)
    time_mat = np.rint((travel + service[:, None]) * TIME_SCALE).astype(np.int64)

    tw_start = np.rint(tw[:, 0] * TIME_SCALE).astype(np.int64)
    tw_end = np.rint(tw[:, 1] * TIME_SCALE).astype(np.int64)

    demand_i = np.rint(signed_demand * CAP_SCALE).astype(np.int64)
    cap_i = int(round(vehicle_capacity * CAP_SCALE))

    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(dist_i[i, j])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def demand_cb(from_index: int) -> int:
        i = manager.IndexToNode(from_index)
        return int(demand_i[i])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        [cap_i] * num_vehicles,
        True,
        "Capacity",
    )

    def time_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(time_mat[i, j])

    time_idx = routing.RegisterTransitCallback(time_cb)
    depot_horizon = int(tw_end[0])
    routing.AddDimension(
        time_idx,
        depot_horizon,
        depot_horizon,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(n_nodes):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(tw_start[node]), int(tw_end[node]))

    for v in range(num_vehicles):
        s = routing.Start(v)
        e = routing.End(v)
        time_dim.CumulVar(s).SetRange(int(tw_start[0]), int(tw_end[0]))
        time_dim.CumulVar(e).SetRange(int(tw_start[0]), int(tw_end[0]))

    solver = routing.solver()
    for p in pickup_idx.tolist():
        d = int(pair_index[p])
        p_idx = manager.NodeToIndex(int(p))
        d_idx = manager.NodeToIndex(d)
        routing.AddPickupAndDelivery(p_idx, d_idx)
        solver.Add(routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx))
        solver.Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.time_limit.seconds = max(1, int(math.ceil(max_runtime)))
    search.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
    )
    search.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
    )

    solution = routing.SolveWithParameters(search)
    if solution is None:
        return _build_fallback_solution(instance)

    actions: list[int] = []
    for vehicle in range(num_vehicles):
        idx = routing.Start(vehicle)
        route_nodes: list[int] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route_nodes.append(node)
            idx = solution.Value(routing.NextVar(idx))

        if route_nodes:
            actions.extend(route_nodes)
            actions.append(0)

    if not actions:
        actions = [0]

    cost = solution.ObjectiveValue() / DIST_SCALE
    return Tensor(actions).long(), torch.tensor(float(cost))


if __name__ == "__main__":
    generator = PDPTWGenerator(num_loc=50)
    td = generator._generate(batch_size=1)

    solution, cost = solve(td)
    #print soultion in routes and cost
    print("Solution:", solution)
    print("Cost:", cost)