import numpy as np

from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria, NoImprovement
from rl4co.envs.routing.mtvrp.baselines.constants import PYVRP_SCALING_FACTOR
from rl4co.envs.routing.mtvrp.baselines.pyvrp import solution2action
from rl4co.envs.routing.mtvrp.baselines.utils import scale
from tensordict.tensordict import TensorDict
from torch import Tensor


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Based on the instance2data from ai4co but adapted for the new version of pyvrp.
    """
    num_locs = instance["demand_backhaul"].size()[0]

    time_windows = scale(instance["time_windows"], scaling_factor)
    pickup = scale(instance["demand_backhaul"], scaling_factor)
    delivery = scale(instance["demand_linehaul"], scaling_factor)
    service = scale(instance["service_time"], scaling_factor)
    coords = scale(instance["locs"], scaling_factor)
    capacity = scale(instance["vehicle_capacity"], scaling_factor)
    max_distance = scale(instance["distance_limit"], scaling_factor)

    depot = Depot(
        x=coords[0][0],
        y=coords[0][1],
    )

    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            tw_early=time_windows[idx][0],
            tw_late=time_windows[idx][1],
            delivery=[delivery[idx]],
            pickup=[pickup[idx]],
            service_duration=service[idx],
            name=f"client_{idx + 1}",
        )
        for idx in range(1, num_locs)
    ]

    vehicle_type = VehicleType(
        num_available=num_locs - 1,  # one vehicle per client
        capacity=[capacity],
        max_distance=max_distance,
        tw_early=time_windows[0][0],
        tw_late=time_windows[0][1],
    )

    matrix = scale(instance["cost_matrix"], scaling_factor)

    if instance["open_route"]:
        matrix[:, 0] = 0

    linehaul = np.flatnonzero(delivery > 0)
    backhaul = np.flatnonzero(pickup > 0)
    matrix[np.ix_(backhaul, linehaul)] = MAX_VALUE

    return ProblemData(clients, [depot], [vehicle_type], [matrix], [matrix])


def solve(
    instance: TensorDict,
    max_runtime: float,
    max_iterations: int = 1000,
    max_iterations_no_improvement: int = 301 * 4,
) -> tuple[Tensor, Tensor]:
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MultipleCriteria(
        [
            MaxRuntime(max_runtime),
            NoImprovement(max_iterations_no_improvement),
            MaxIterations(max_iterations),
        ]
    )
    result = _solve(data, stop)
    solution = result.best
    action = solution2action(solution)
    cost = -result.cost() / PYVRP_SCALING_FACTOR

    return action, cost  # type: ignore[return-value]
