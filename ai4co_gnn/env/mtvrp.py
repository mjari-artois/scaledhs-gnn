import os

from functools import partial
from typing import Optional

import numpy as np
import torch

from rl4co.envs.routing.mtvrp.baselines.utils import process_instance
from rl4co.envs.routing.mtvrp.env import MTVRPEnv as MTVRPEnvBase
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.utils import get_pylogger
from rl4co.utils.ops import batchify, gather_by_index, get_distance
from tensordict import TensorDict
from torch import Tensor

from ai4co_gnn.data.cada import load_cada_npz

log = get_pylogger(__name__)


class MTVRPEnv(MTVRPEnvBase):
    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        check_solution: bool = False,
        use_recourse_for_infeasible: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        **kwargs,
    ):
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            check_solution=check_solution,
            **kwargs,
        )
        self.use_recourse_for_infeasible = use_recourse_for_infeasible
        if self.use_recourse_for_infeasible and self.check_solution:
            log.warning(
                "Disabling check_solution because recourse mode allows infeasible actions."
            )
            self.check_solution = False
        self.load_solutions = load_solutions
        self.solution_fname = solution_fname

    def _step(self, td: TensorDict) -> TensorDict:
        if not self.use_recourse_for_infeasible:
            return super()._step(td)

        action = td["action"]
        is_depot = action == 0

        # In recourse mode, td["action_mask"] may be served-only.
        # Recompute full constraints here for post-decision feasibility.
        feasible = self._get_constraint_action_mask(td).gather(-1, action[..., None])
        infeasible_customer = (~is_depot[..., None]) & (~feasible)

        # Save vehicle state before stepping
        saved = {
            k: td[k].clone()
            for k in [
                "current_node",
                "current_time",
                "current_route_length",
                "used_capacity_linehaul",
                "used_capacity_backhaul",
            ]
        }

        td = super()._step(td)

        # Revert vehicle state for infeasible customers (visited=True stays as recourse)
        for k, v in saved.items():
            if k == "current_node":
                td[k] = torch.where(infeasible_customer.squeeze(-1), v, td[k])
            else:
                td[k] = torch.where(infeasible_customer, v, td[k])

        td.set("action_mask", self.get_action_mask(td))
        return td

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        if not self.use_recourse_for_infeasible:
            return MTVRPEnvBase.get_action_mask(td)
        return self._served_only_action_mask(td)

    @staticmethod
    def _served_only_action_mask(td: TensorDict) -> torch.Tensor:
        visited = td["visited"]
        mask = ~visited.clone()
        mask[..., 0] = True

        has_unserved_customers = (~visited[..., 1:]).any(dim=-1)
        at_depot = td["current_node"] == 0
        mask[..., 0] = mask[..., 0] & ~(at_depot & has_unserved_customers)
        return mask

    @staticmethod
    def _get_constraint_action_mask(td: TensorDict) -> torch.Tensor:
        return MTVRPEnvBase.get_action_mask(td)

    @staticmethod
    def complete_data(td):
        # backwards compatibility
        if "open_route" not in td.keys():
            td.set(
                "open_route",
                torch.zeros_like(td["vehicle_capacity"], dtype=torch.bool),
            )

        if "demand_backhaul" not in td.keys():
            td.set("demand_backhaul", torch.zeros_like(td["demand_linehaul"]))

        if td["demand_linehaul"].shape[-1] < td["locs"].shape[-2]:
            # append zero demand for depot
            td.set(
                "demand_linehaul",
                torch.cat(
                    [
                        torch.zeros_like(td["demand_linehaul"][..., :1]),
                        td["demand_linehaul"],
                    ],
                    dim=-1,
                ),
            )
            td.set(
                "demand_backhaul",
                torch.cat(
                    [
                        torch.zeros_like(td["demand_backhaul"][..., :1]),
                        td["demand_backhaul"],
                    ],
                    dim=-1,
                ),
            )

        if "time_windows" not in td.keys():
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
            td.set("time_windows", time_windows)

        if "service_time" not in td.keys():
            td.set("service_time", torch.zeros_like(td["demand_linehaul"]))

        if "distance_limit" not in td.keys():
            td.set(
                "distance_limit",
                torch.full_like(td["vehicle_capacity"], float("inf")),
            )

        if "capacity_original" not in td.keys():
            td.set("capacity_original", td["vehicle_capacity"].clone())

        return td

    def load_data(self, fpath, batch_size=[], scale=False):
        if "CADA/synthetic_data" in str(fpath):
            td_load = load_cada_npz(fpath)
        else:
            td_load = super().load_data(fpath, batch_size=batch_size)
        self.complete_data(td_load)
        if self.load_solutions:
            solution_fpath = fpath.replace(".npz", self.solution_fname)
            if os.path.exists(solution_fpath):
                sol = np.load(solution_fpath)
                sol_dict = {}
                for key, value in sol.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0:
                        if value.shape[0] == td_load.batch_size[0]:
                            key = "costs_bks" if key == "costs" else key
                            key = "actions_bks" if key == "actions" else key
                            sol_dict[key] = torch.tensor(value)
                td_load.update(sol_dict)
            else:
                log.warning(f"No solution file found at {solution_fpath}")
        return td_load

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        td = self.complete_data(td)

        costs_bks = td.get("costs_bks", None)
        actions_bks = td.get("actions_bks", None)

        td = super()._reset(td, batch_size)

        td.set("costs_bks", costs_bks)
        td.set("actions_bks", actions_bks)

        return td

    def _get_reward(self, td: TensorDict, actions: Tensor) -> Tensor:
        if not self.use_recourse_for_infeasible:
            return super()._get_reward(td, actions)
        return self._get_reward_with_recourse(td, actions)

    def get_recourse_stats(self, td: TensorDict, actions: Tensor) -> dict[str, Tensor]:
        """Return per-instance recourse statistics for a decoded action sequence."""
        if actions.dim() > 2:
            actions = actions.reshape(-1, actions.size(-1))

        td_batch = td.batch_size.numel()
        action_batch = actions.size(0)
        if td_batch != action_batch:
            if action_batch % td_batch != 0:
                raise RuntimeError(
                    f"Incompatible batch sizes between td ({td_batch}) and actions ({action_batch})."
                )
            td = batchify(td, action_batch // td_batch)

        (
            _,
            recourse_cost,
            recourse_count,
            customer_actions,
        ) = self._simulate_actions_with_recourse(td, actions)
        recourse_rate = recourse_count / customer_actions.clamp_min(1.0)
        return {
            "recourse_cost": recourse_cost,
            "recourse_count": recourse_count,
            "customer_actions": customer_actions,
            "recourse_rate": recourse_rate,
        }

    def _simulate_actions_with_recourse(
        self, td: TensorDict, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Replay actions with post-decision feasibility checks and recourse accounting."""
        batch_size = actions.size(0)
        device = td.device
        float_dtype = td["locs"].dtype

        locs = td["locs"]
        demand_linehaul = td["demand_linehaul"]
        demand_backhaul = td["demand_backhaul"]
        service_time = td["service_time"]
        time_windows = td["time_windows"]
        vehicle_capacity = td["vehicle_capacity"].squeeze(-1)
        speed = td["speed"].squeeze(-1)
        distance_limit = td["distance_limit"].squeeze(-1)
        open_route = td["open_route"].squeeze(-1)
        depot_loc = locs[:, 0]
        depot_late_tw = time_windows[:, 0, 1]

        curr_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        curr_time = torch.zeros(batch_size, dtype=float_dtype, device=device)
        curr_route_length = torch.zeros(batch_size, dtype=float_dtype, device=device)
        used_capacity_linehaul = torch.zeros(batch_size, dtype=float_dtype, device=device)
        used_capacity_backhaul = torch.zeros(batch_size, dtype=float_dtype, device=device)
        visited = torch.zeros_like(demand_linehaul, dtype=torch.bool)
        total_cost = torch.zeros(batch_size, dtype=float_dtype, device=device)
        recourse_cost_total = torch.zeros(batch_size, dtype=float_dtype, device=device)
        recourse_count = torch.zeros(batch_size, dtype=float_dtype, device=device)
        customer_actions = torch.zeros(batch_size, dtype=float_dtype, device=device)

        for step in range(actions.size(1)):
            next_node = actions[:, step]
            is_depot = next_node == 0

            curr_loc = gather_by_index(locs, curr_node)
            next_loc = gather_by_index(locs, next_node)

            dist_to_next = get_distance(curr_loc, next_loc)
            dist_next_to_depot = get_distance(next_loc, depot_loc)
            dist_depot_to_next = get_distance(depot_loc, next_loc)

            selected_tw = gather_by_index(time_windows, next_node)
            selected_service = gather_by_index(service_time, next_node)
            selected_linehaul = gather_by_index(demand_linehaul, next_node)
            selected_backhaul = gather_by_index(demand_backhaul, next_node)
            selected_visited = gather_by_index(visited, next_node)

            arrival_time = curr_time + dist_to_next / speed
            can_reach_customer = arrival_time < selected_tw[:, 1]
            return_deadline = (
                torch.max(arrival_time, selected_tw[:, 0])
                + selected_service
                + dist_next_to_depot / speed
            )
            can_reach_depot = open_route | (return_deadline < depot_late_tw)

            exceeds_dist_limit = (
                curr_route_length + dist_to_next + dist_next_to_depot * ~open_route
                > distance_limit
            )

            linehauls_missing = (demand_linehaul * ~visited).sum(-1) > 0
            is_carrying_backhaul = gather_by_index(demand_backhaul, curr_node) > 0

            exceeds_cap_linehaul = (
                selected_linehaul + used_capacity_linehaul > vehicle_capacity
            )
            exceeds_cap_backhaul = (
                selected_backhaul + used_capacity_backhaul > vehicle_capacity
            )
            meets_demand_constraint = (
                linehauls_missing
                & ~exceeds_cap_linehaul
                & ~is_carrying_backhaul
                & (selected_linehaul > 0)
            ) | (~exceeds_cap_backhaul & (selected_backhaul > 0))

            feasible_customer = (
                ~is_depot
                & ~selected_visited
                & can_reach_customer
                & can_reach_depot
                & meets_demand_constraint
                & ~exceeds_dist_limit
            )

            infeasible_customer = (~is_depot) & ~feasible_customer
            recourse_trip_cost = dist_depot_to_next + dist_next_to_depot * ~open_route
            recourse_step_cost = recourse_trip_cost * infeasible_customer

            total_cost = (
                total_cost
                + dist_to_next * feasible_customer
                + recourse_step_cost
                + dist_to_next * is_depot * ~open_route
            )
            recourse_cost_total = recourse_cost_total + recourse_step_cost
            recourse_count = recourse_count + infeasible_customer.to(float_dtype)
            customer_actions = customer_actions + (~is_depot).to(float_dtype)

            curr_node = torch.where(feasible_customer | is_depot, next_node, curr_node)
            new_time = torch.max(arrival_time, selected_tw[:, 0]) + selected_service
            curr_time = torch.where(feasible_customer, new_time, curr_time)
            curr_time = torch.where(is_depot, torch.zeros_like(curr_time), curr_time)
            curr_route_length = torch.where(
                feasible_customer, curr_route_length + dist_to_next, curr_route_length
            )
            curr_route_length = torch.where(
                is_depot, torch.zeros_like(curr_route_length), curr_route_length
            )
            used_capacity_linehaul = torch.where(
                feasible_customer,
                used_capacity_linehaul + selected_linehaul,
                used_capacity_linehaul,
            )
            used_capacity_linehaul = torch.where(
                is_depot, torch.zeros_like(used_capacity_linehaul), used_capacity_linehaul
            )
            used_capacity_backhaul = torch.where(
                feasible_customer,
                used_capacity_backhaul + selected_backhaul,
                used_capacity_backhaul,
            )
            used_capacity_backhaul = torch.where(
                is_depot, torch.zeros_like(used_capacity_backhaul), used_capacity_backhaul
            )
            visited = visited.scatter(-1, next_node.unsqueeze(-1), True)

        final_dist_to_depot = get_distance(gather_by_index(locs, curr_node), depot_loc)
        total_cost = total_cost + final_dist_to_depot * ~open_route

        return total_cost, recourse_cost_total, recourse_count, customer_actions

    def _get_reward_with_recourse(self, td: TensorDict, actions: Tensor) -> Tensor:
        """Compute cost with post-decision feasibility checks and recourse routes."""
        total_cost, _, _, _ = self._simulate_actions_with_recourse(td, actions)
        return -total_cost

    @staticmethod
    def solve(
        instances: TensorDict,
        max_runtime: float,
        num_procs: int = 1,
        solver: str = "pyvrp",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        instances = process_instance(instances)

        try:
            from ai4co_gnn.baselines import pyvrp
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Solver 'pyvrp' requires the optional dependency `pyvrp`."
            ) from error

        assert (
            solver == "pyvrp"
        ), "Only 'pyvrp' solver is currently supported in MTVRPEnv."

        func = partial(pyvrp.solve, max_runtime=max_runtime, **kwargs)

        if num_procs > 1:
            results = []
            from pebble import ProcessPool

            with ProcessPool(max_workers=num_procs) as pool:
                # The 'map' method handles everything, including the timeout.
                future = pool.map(func, instances, timeout=max_runtime)

                iterator = future.result()

                while True:
                    try:
                        result = next(iterator)
                        results.append(result)
                    except StopIteration:
                        break  # All tasks finished
                    except TimeoutError:
                        log.warning(
                            f"A task was killed because it exceeded the {max_runtime}s timeout."
                        )
                        # Pebble handles process termination automatically.
                        # You can append a default result if you wish.
                        results.append(([], float("-inf")))
                    except Exception as error:
                        log.error(f"A task failed with an exception: {error}")
                        results.append(([], float("-inf")))

        else:
            results = [func(instance) for instance in instances]

        actions, costs = zip(*results)

        max_len = max(len(action) for action in actions)
        padded_actions = [action + [0] * (max_len - len(action)) for action in actions]

        return Tensor(padded_actions).long(), Tensor(costs)
