import os

from functools import partial
from typing import Optional

import numpy as np
import torch

from rl4co.envs.routing.mtvrp.baselines.utils import process_instance
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils import get_pylogger
from rl4co.utils.ops import batchify, gather_by_index, get_distance
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded

from .generator import PDPTWGenerator

log = get_pylogger(__name__)


class PDPTWEnv(RL4COEnvBase):
    """Pickup-and-Delivery VRP with Time Windows, recourse-mode.

    Each request is a pickup/delivery pair. The policy emits a sequence of node
    indices; the env scores it as ``main_route_distance + Σ recourse_trip_costs``.

    Recourse rule (see plan ``we-need-to-work-precious-abelson.md`` §"Recourse rule"):
        - The policy mask is served-only (``~visited`` plus the standard depot
          suppression). Time windows, capacity, and precedence are NOT in the
          policy mask — the model learns them from the cost signal.
        - Trigger A: chosen customer fails the internal feasibility predicate
          (TW / capacity / precedence) → the whole pair is served by a recourse
          trip ``depot → pickup → delivery → depot`` and both nodes are marked
          ``visited``; vehicle main-route state is reverted.
        - Trigger B: action is the depot and there exist pickups in the
          current route whose deliveries are still unserved → one recourse
          trip per such pair.
        - End-of-episode safety net: any unserved pair gets a recourse trip
          (should be empty under the served-only mask, kept as a guard).
    """

    name = "pdptw"

    def __init__(
        self,
        generator: PDPTWGenerator = None,
        generator_params: dict = {},
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        use_recourse_for_infeasible: bool = True,
        **kwargs,
    ):
        # Strict (non-recourse) mode is unsupported for PDPTW: the served-only
        # mask is the only safe mask. A full-constraint mask can deadlock when
        # picking a pickup orphans its delivery (TW closes mid-route, no other
        # customer fits, depot return cannot recover). The flag is kept for
        # backward compatibility and config parity with the MTVRP env, but
        # values other than True raise.
        if not use_recourse_for_infeasible:
            raise ValueError(
                "PDPTWEnv only supports recourse mode "
                "(use_recourse_for_infeasible=True). Strict mode can deadlock "
                "when a pickup is served and its delivery becomes unreachable."
            )
        super().__init__(check_solution=check_solution, **kwargs)
        if generator is None:
            generator = PDPTWGenerator(**generator_params)
        self.generator = generator
        self.use_recourse_for_infeasible = True
        if self.check_solution:
            log.warning(
                "Disabling check_solution because recourse mode allows infeasible actions."
            )
            self.check_solution = False
        self.load_solutions = load_solutions
        self.solution_fname = solution_fname
        self._make_spec(self.generator)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def complete_data(td: TensorDict) -> TensorDict:
        """Backfill td keys that may be missing when loading old npz files."""
        n_total = td["locs"].shape[-2]
        n_req = (n_total - 1) // 2

        if "pair_index" not in td.keys():
            pair_index = torch.zeros(
                *td.batch_size, n_total, dtype=torch.long, device=td.device
            )
            pickup_idx = torch.arange(1, n_req + 1, dtype=torch.long, device=td.device)
            delivery_idx = torch.arange(
                n_req + 1, n_total, dtype=torch.long, device=td.device
            )
            pair_index[..., 1 : n_req + 1] = delivery_idx
            pair_index[..., n_req + 1 :] = pickup_idx
            td.set("pair_index", pair_index)

        if "is_pickup" not in td.keys():
            is_pickup = torch.zeros(
                *td.batch_size, n_total, dtype=torch.bool, device=td.device
            )
            is_pickup[..., 1 : n_req + 1] = True
            td.set("is_pickup", is_pickup)

        if "is_delivery" not in td.keys():
            is_delivery = torch.zeros(
                *td.batch_size, n_total, dtype=torch.bool, device=td.device
            )
            is_delivery[..., n_req + 1 :] = True
            td.set("is_delivery", is_delivery)

        if "pickup_idx" not in td.keys():
            pickup_idx = torch.arange(1, n_req + 1, dtype=torch.long, device=td.device)
            td.set(
                "pickup_idx",
                pickup_idx.view(*([1] * len(td.batch_size)), n_req).expand(
                    *td.batch_size, n_req
                ),
            )

        if "delivery_idx" not in td.keys():
            delivery_idx = torch.arange(
                n_req + 1, n_total, dtype=torch.long, device=td.device
            )
            td.set(
                "delivery_idx",
                delivery_idx.view(*([1] * len(td.batch_size)), n_req).expand(
                    *td.batch_size, n_req
                ),
            )

        if "capacity_original" not in td.keys():
            td.set("capacity_original", td["vehicle_capacity"].clone())

        if "demand_linehaul" not in td.keys():
            source = td.get("delivery_demand", None)
            if source is None:
                raise KeyError(
                    "PDPTWEnv requires either `demand_linehaul` or "
                    "`delivery_demand` in the input TensorDict."
                )
            demand_linehaul = torch.zeros(
                *td.batch_size, n_total, dtype=source.dtype, device=td.device
            )
            demand_linehaul[..., n_req + 1 :] = source[..., n_req + 1 :]
            td.set("demand_linehaul", demand_linehaul)

        if "demand_backhaul" not in td.keys():
            source = td.get("pickup_demand", None)
            if source is None:
                raise KeyError(
                    "PDPTWEnv requires either `demand_backhaul` or "
                    "`pickup_demand` in the input TensorDict."
                )
            demand_backhaul = torch.zeros(
                *td.batch_size, n_total, dtype=source.dtype, device=td.device
            )
            demand_backhaul[..., 1 : n_req + 1] = source[..., 1 : n_req + 1]
            td.set("demand_backhaul", demand_backhaul)

        return td

    def load_data(self, fpath, batch_size=[], scale=False):
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

    # ------------------------------------------------------------------ reset/step

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        td = self.complete_data(td)
        device = td.device
        n = td["locs"].shape[-2]

        costs_bks = td.get("costs_bks", None)
        actions_bks = td.get("actions_bks", None)

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand_backhaul": td["demand_backhaul"],
                "demand_linehaul": td["demand_linehaul"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "vehicle_capacity": td["vehicle_capacity"],
                "capacity_original": td["capacity_original"],
                "speed": td["speed"],
                "pair_index": td["pair_index"],
                "is_pickup": td["is_pickup"],
                "is_delivery": td["is_delivery"],
                "pickup_idx": td["pickup_idx"],
                "delivery_idx": td["delivery_idx"],
                # ``open_route`` and ``distance_limit`` are not real PDPTW
                # constraints but are kept in td for compatibility with the
                # MTVRP-aligned init/context embeddings. They are read by
                # AttentionAndDistanceModelDecoder (open_route) and by the
                # additive-backbone init embedding (distance_limit), which
                # both treat the dummy values as inert (closed route +
                # ``nan_to_num(posinf=0)`` on inf).
                "open_route": torch.zeros(
                    (*batch_size, 1), dtype=torch.bool, device=device
                ),
                "distance_limit": torch.full(
                    (*batch_size, 1), float("inf"), dtype=torch.float32, device=device
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "current_load": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                # MTVRP-compatible state slots used by the transferred
                # LB-to-PDPTW decoder context. These are auxiliary route
                # counters only; PDPTW feasibility still uses ``current_load``.
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, n), dtype=torch.bool, device=device
                ),
                "in_current_route": torch.zeros(
                    (*batch_size, n), dtype=torch.bool, device=device
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        if costs_bks is not None:
            td_reset.set("costs_bks", costs_bks)
        if actions_bks is not None:
            td_reset.set("actions_bks", actions_bks)
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """Advance the environment by one action.

        Policy mask (:meth:`_served_only_action_mask`) only forbids revisits, so
        any non-depot action arrives here. We then check the chosen node against
        TW / capacity / precedence and:
            - if feasible: advance vehicle state along the main route;
            - if depot: reset main-route state (start a new route);
            - otherwise (infeasible customer): leave vehicle state unchanged
              and mark the pair as visited (Trigger A — cost charged later in
              the recourse simulator).

        Trigger B fires when the action is the depot and there are pickups
        already served on the main route whose deliveries are still unserved:
        each such pair is marked visited; the depot return clears
        ``in_current_route`` for every node so it cannot re-fire on the next
        depot return.
        """
        action = td["action"]  # [B]
        is_depot = action == 0

        # ---- gather selected-node features once ----
        locs = td["locs"]
        speed = td["speed"]
        cap = td["vehicle_capacity"]
        pair_index = td["pair_index"]

        curr_loc = gather_by_index(locs, td["current_node"])
        next_loc = gather_by_index(locs, action)
        d_ij = get_distance(curr_loc, next_loc)[..., None]  # [B, 1]
        d_j0 = get_distance(next_loc, locs[..., 0, :])[..., None]  # [B, 1]
        sel_tw = gather_by_index(td["time_windows"], action, dim=1, squeeze=False)
        sel_early, sel_late = sel_tw[..., 0], sel_tw[..., 1]
        sel_svc = gather_by_index(td["service_time"], action, dim=1, squeeze=False)
        sel_p_d = gather_by_index(td["demand_backhaul"], action, dim=1, squeeze=False)
        sel_d_d = gather_by_index(td["demand_linehaul"], action, dim=1, squeeze=False)
        is_p = gather_by_index(td["is_pickup"], action, dim=1, squeeze=False)
        is_d = gather_by_index(td["is_delivery"], action, dim=1, squeeze=False)
        sel_visited = gather_by_index(td["visited"], action, dim=1, squeeze=False)
        pair = gather_by_index(pair_index, action, dim=1, squeeze=False)
        pair_visited = td["visited"].gather(-1, pair)
        pair_in_current_route = td["in_current_route"].gather(-1, pair)

        # ---- post-decision feasibility (TW + capacity + precedence) ----
        depot_late_tw = td["time_windows"][:, 0:1, 1]
        arrival = td["current_time"] + d_ij / speed
        can_reach_customer = arrival < sel_late
        can_reach_depot = (
            torch.maximum(arrival, sel_early) + sel_svc + d_j0 / speed
        ) < depot_late_tw
        cap_ok = (is_p & (td["current_load"] + sel_p_d <= cap)) | (
            is_d & (td["current_load"] >= sel_d_d)
        )
        # Delivery only allowed if its pickup is in the current route — i.e.
        # the vehicle physically carries the load right now. After a depot
        # return the load is reset, so a delivery whose pickup was visited in
        # an earlier route is no longer servable on the main tour.
        prec_ok = is_p | (is_d & pair_visited & pair_in_current_route)
        feasible = (
            ~sel_visited
            & can_reach_customer
            & can_reach_depot
            & cap_ok
            & prec_ok
        )  # [B, 1]

        is_depot_b = is_depot[..., None]
        feas_cust_b = (~is_depot_b) & feasible
        infeas_cust_b = (~is_depot_b) & ~feasible

        # ---- vehicle state updates ----
        new_time_customer = torch.maximum(arrival, sel_early) + sel_svc
        new_load_customer = (
            td["current_load"]
            + sel_p_d * is_p.to(td["current_load"].dtype)
            - sel_d_d * is_d.to(td["current_load"].dtype)
        )
        new_used_capacity_linehaul = (
            td["used_capacity_linehaul"]
            + sel_d_d * is_d.to(td["used_capacity_linehaul"].dtype)
        )
        new_used_capacity_backhaul = (
            td["used_capacity_backhaul"]
            + sel_p_d * is_p.to(td["used_capacity_backhaul"].dtype)
        )
        new_route_len_customer = td["current_route_length"] + d_ij

        zeros_t = torch.zeros_like(td["current_time"])
        zeros_load = torch.zeros_like(td["current_load"])
        zeros_rl = torch.zeros_like(td["current_route_length"])
        zeros_used_linehaul = torch.zeros_like(td["used_capacity_linehaul"])
        zeros_used_backhaul = torch.zeros_like(td["used_capacity_backhaul"])

        new_current_time = torch.where(
            feas_cust_b,
            new_time_customer,
            torch.where(is_depot_b, zeros_t, td["current_time"]),
        )
        new_current_load = torch.where(
            feas_cust_b,
            new_load_customer,
            torch.where(is_depot_b, zeros_load, td["current_load"]),
        )
        new_current_route_length = torch.where(
            feas_cust_b,
            new_route_len_customer,
            torch.where(is_depot_b, zeros_rl, td["current_route_length"]),
        )
        new_used_capacity_linehaul = torch.where(
            feas_cust_b,
            new_used_capacity_linehaul,
            torch.where(
                is_depot_b, zeros_used_linehaul, td["used_capacity_linehaul"]
            ),
        )
        new_used_capacity_backhaul = torch.where(
            feas_cust_b,
            new_used_capacity_backhaul,
            torch.where(
                is_depot_b, zeros_used_backhaul, td["used_capacity_backhaul"]
            ),
        )
        new_current_node = torch.where(
            feas_cust_b.squeeze(-1) | is_depot, action, td["current_node"]
        )

        # ---- visited / in_current_route updates ----
        visited = td["visited"]
        in_current_route = td["in_current_route"]
        action_idx = action[..., None]  # [B, 1]

        # Mark action visited (any non-depot action — feasible OR Trigger A).
        visited = visited.scatter(
            -1, action_idx, visited.gather(-1, action_idx) | (~is_depot_b)
        )
        # Trigger A: partner of an infeasible customer joins the recourse trip.
        visited = visited.scatter(
            -1, pair, visited.gather(-1, pair) | infeas_cust_b
        )
        # ``in_current_route`` flips True at the action only when the customer
        # was served on the main route (vehicle physically carries this pair's
        # state until the next depot return).
        in_current_route = in_current_route.scatter(
            -1, action_idx, in_current_route.gather(-1, action_idx) | feas_cust_b
        )


        # Trigger B: at depot return, every pickup still in the current route
        # whose delivery is unserved triggers a recourse trip; the delivery is
        # marked visited.
        pair_visited_per_node = visited.gather(-1, pair_index)
        stranded = td["is_pickup"] & in_current_route & ~pair_visited_per_node
        trigger_B = is_depot_b & stranded  # [B, N]
        delivery_update = torch.zeros_like(visited, dtype=torch.long)
        delivery_update.scatter_add_(-1, pair_index, trigger_B.long())
        visited = visited | (delivery_update > 0)

        # Depot return ends the current route — clear ``in_current_route``
        # for the whole row. This also subsumes the Trigger B clear.
        in_current_route = in_current_route & ~is_depot_b

        done = visited[..., 1:].all(-1)
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "current_node": new_current_node,
                "current_time": new_current_time,
                "current_load": new_current_load,
                "current_route_length": new_current_route_length,
                "used_capacity_linehaul": new_used_capacity_linehaul,
                "used_capacity_backhaul": new_used_capacity_backhaul,
                "visited": visited,
                "in_current_route": in_current_route,
                "done": done,
                "reward": reward,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    # ------------------------------------------------------------------ mask

    def get_action_mask(self, td: TensorDict) -> Tensor:
        return self._served_only_action_mask(td)

    @staticmethod
    def _served_only_action_mask(td: TensorDict) -> Tensor:
        """Recourse-mode policy mask: only ``~visited`` plus depot suppression.

        The full set of constraints (TW, capacity, precedence) is intentionally
        NOT in the policy mask — strict masking can deadlock when picking a
        pickup orphans its delivery. The model learns these constraints from
        the recourse cost signal instead.
        """
        visited = td["visited"]
        mask = ~visited.clone()
        mask[..., 0] = True
        has_unserved = (~visited[..., 1:]).any(dim=-1)
        at_depot = td["current_node"] == 0
        mask[..., 0] = mask[..., 0] & ~(at_depot & has_unserved)
        return mask

    # ------------------------------------------------------------------ reward

    def _get_reward(self, td: TensorDict, actions: Tensor) -> Tensor:
        total_cost, _, _, _ = self._simulate_actions_with_recourse(td, actions)
        return -total_cost

    def get_recourse_stats(
        self, td: TensorDict, actions: Tensor
    ) -> dict[str, Tensor]:
        """Per-instance recourse statistics for a decoded action sequence."""
        if actions.dim() > 2:
            actions = actions.reshape(-1, actions.size(-1))

        td_batch = td.batch_size.numel()
        action_batch = actions.size(0)
        if td_batch != action_batch:
            if action_batch % td_batch != 0:
                raise RuntimeError(
                    f"Incompatible batch sizes between td ({td_batch}) and "
                    f"actions ({action_batch})."
                )
            td = batchify(td, action_batch // td_batch)

        total_cost, recourse_cost, recourse_count, customer_actions, violation_stats = (
            self._simulate_actions_with_recourse(
                td,
                actions,
                return_violation_stats=True,
            )
        )

        recourse_rate = recourse_count / customer_actions.clamp_min(1.0)

        out = {
            "total_cost": total_cost,
            "recourse_cost": recourse_cost,
            "recourse_count": recourse_count,
            "customer_actions": customer_actions,
            "recourse_rate": recourse_rate,
        }

        out.update(violation_stats)
        return out

    def _simulate_actions_with_recourse(
        self, td: TensorDict, actions: Tensor, return_violation_stats = False
    ):
        """Replay actions with post-decision feasibility checks and recourse accounting."""
        B = actions.size(0)
        T = actions.size(1)
        device = td.device
        float_dtype = td["locs"].dtype

        locs = td["locs"]
        pickup_demand = td["demand_backhaul"]
        delivery_demand = td["demand_linehaul"]
        pair_index = td["pair_index"]
        is_pickup_t = td["is_pickup"]
        is_delivery_t = td["is_delivery"]
        cap = td["vehicle_capacity"].squeeze(-1)
        speed = td["speed"].squeeze(-1)
        tw = td["time_windows"]
        svc_t = td["service_time"]
        depot_loc = locs[:, 0]
        depot_late_tw = tw[:, 0, 1]

        # Per-pair recourse trip cost (depot → pickup_node → delivery_node → depot),
        # indexed by *node*. For deliveries, this would be d→p→d→0 — wrong direction;
        # we never read this slot for deliveries, only for pickups (Trigger B uses
        # ``is_pickup_t`` mask, end-of-episode uses ``pickup_unserved``).
        delivery_locs_per_node = torch.gather(
            locs, 1, pair_index.unsqueeze(-1).expand(-1, -1, 2)
        )
        d_0_p_full = torch.norm(locs - depot_loc.unsqueeze(1), dim=-1)
        d_p_d_full = torch.norm(locs - delivery_locs_per_node, dim=-1)
        d_d_0_full = torch.norm(delivery_locs_per_node - depot_loc.unsqueeze(1), dim=-1)
        per_pickup_recourse_cost = d_0_p_full + d_p_d_full + d_d_0_full  # [B, N]

        curr_node = torch.zeros(B, dtype=torch.long, device=device)
        curr_time = torch.zeros(B, dtype=float_dtype, device=device)
        curr_load = torch.zeros(B, dtype=float_dtype, device=device)
        visited = torch.zeros((B, locs.size(1)), dtype=torch.bool, device=device)
        in_current_route = torch.zeros_like(visited)
        total_cost = torch.zeros(B, dtype=float_dtype, device=device)
        recourse_cost = torch.zeros(B, dtype=float_dtype, device=device)
        recourse_count = torch.zeros(B, dtype=float_dtype, device=device)
        customer_actions = torch.zeros(B, dtype=float_dtype, device=device)

        capacity_violation_count = torch.zeros(B, dtype=float_dtype, device=device)
        time_window_violation_count = torch.zeros(B, dtype=float_dtype, device=device)
        precedence_violation_count = torch.zeros(B, dtype=float_dtype, device=device)
        open_pickup_violation_count = torch.zeros(B, dtype=float_dtype, device=device)

        for step in range(T):
            nxt = actions[:, step]
            is_depot = nxt == 0

            is_p = gather_by_index(is_pickup_t, nxt)
            is_d = gather_by_index(is_delivery_t, nxt)
            pair = gather_by_index(pair_index, nxt)
            sel_p_d = gather_by_index(pickup_demand, nxt)
            sel_d_d = gather_by_index(delivery_demand, nxt)
            sel_visited = gather_by_index(visited, nxt)
            pair_visited = gather_by_index(visited, pair)
            pair_in_current_route = gather_by_index(in_current_route, pair)

            curr_loc = gather_by_index(locs, curr_node)
            nxt_loc = gather_by_index(locs, nxt)
            d_ij = get_distance(curr_loc, nxt_loc)
            d_j0 = get_distance(nxt_loc, depot_loc)
            sel_tw = gather_by_index(tw, nxt)
            sel_early, sel_late = sel_tw[..., 0], sel_tw[..., 1]
            sel_svc = gather_by_index(svc_t, nxt)

            arrival = curr_time + d_ij / speed
            can_reach_customer = arrival < sel_late
            can_reach_depot = (
                torch.maximum(arrival, sel_early) + sel_svc + d_j0 / speed
            ) < depot_late_tw
            cap_ok_p = is_p & (curr_load + sel_p_d <= cap)
            cap_ok_d = is_d & (curr_load >= sel_d_d)
            capacity_ok = cap_ok_p | cap_ok_d
            prec_ok = is_p | (is_d & pair_visited & pair_in_current_route)

            feasible = (
                (~is_depot)
                & (~sel_visited)
                & can_reach_customer
                & can_reach_depot
                & capacity_ok
                & prec_ok
            )
            infeasible = (~is_depot) & ~feasible

            # Violation counts are meant to explain recourse-causing customer
            # choices, not raw predicate failures on depot or feasible steps.
            capacity_violation_count = capacity_violation_count + (
                (infeasible & ~capacity_ok).to(float_dtype)
            )
            time_window_violation_count = time_window_violation_count + (
                (infeasible & ~(can_reach_customer & can_reach_depot)).to(float_dtype)
            )
            precedence_violation_count = precedence_violation_count + (
                (infeasible & is_d & ~prec_ok).to(float_dtype)
            )
            open_pickup_violation_count = open_pickup_violation_count + (
                (infeasible & is_d & ~pair_visited).to(float_dtype)
            )

            # Trigger A — pair recourse for infeasible customer pick.
            pickup_of_pair = torch.where(is_p, nxt, pair)
            delivery_of_pair = torch.where(is_p, pair, nxt)
            loc_p = gather_by_index(locs, pickup_of_pair)
            loc_d = gather_by_index(locs, delivery_of_pair)
            pair_recourse = (
                get_distance(depot_loc, loc_p)
                + get_distance(loc_p, loc_d)
                + get_distance(loc_d, depot_loc)
            )
            # Defensive guard: the served-only mask already prevents revisits,
            # so ``sel_visited`` should always be False here. Pair-of-action's
            # visited status is intentionally NOT used: if the pickup was
            # served on the main route but the delivery is now infeasible
            # (e.g. capacity already drained on another pair, TW closed), the
            # delivery still mints a full pair recourse trip per the project
            # spec — the model double-pays for the pickup, which is what
            # penalises that mistake.
            already_recoursed = sel_visited
            cost_A = pair_recourse * (infeasible & ~already_recoursed).to(float_dtype)
            n_A = (infeasible & ~already_recoursed).to(float_dtype)

            # Trigger B — at depot, mint trip for every stranded pickup.
            pair_visited_per_node = visited.gather(-1, pair_index)
            stranded = is_pickup_t & in_current_route & ~pair_visited_per_node
            trigger_B = is_depot[..., None] & stranded
            cost_B = (per_pickup_recourse_cost * trigger_B.to(float_dtype)).sum(-1)
            n_B = trigger_B.to(float_dtype).sum(-1)

            # Main-route step cost: closed-route, charge d(curr,next) for feasible
            # customer picks and depot returns. Infeasible customer picks contribute
            # only the pair recourse cost.
            cost_main = d_ij * feasible.to(float_dtype) + d_ij * is_depot.to(float_dtype)

            total_cost = total_cost + cost_main + cost_A + cost_B
            recourse_cost = recourse_cost + cost_A + cost_B
            recourse_count = recourse_count + n_A + n_B
            customer_actions = customer_actions + (~is_depot).to(float_dtype)

            # State updates
            new_time = torch.maximum(arrival, sel_early) + sel_svc
            curr_time = torch.where(feasible, new_time, curr_time)
            curr_time = torch.where(is_depot, torch.zeros_like(curr_time), curr_time)
            delta_load = sel_p_d * is_p.to(float_dtype) - sel_d_d * is_d.to(float_dtype)
            curr_load = torch.where(feasible, curr_load + delta_load, curr_load)
            curr_load = torch.where(is_depot, torch.zeros_like(curr_load), curr_load)
            curr_node = torch.where(feasible | is_depot, nxt, curr_node)

            # visited: action position
            visited_action_was = gather_by_index(visited, nxt)
            visited = visited.scatter(
                -1, nxt[..., None], (visited_action_was | ~is_depot)[..., None]
            )
            # visited: partner of infeasible customer
            pair_was = gather_by_index(visited, pair)
            visited = visited.scatter(
                -1, pair[..., None], (pair_was | infeasible)[..., None]
            )
            # visited: deliveries flagged by Trigger B
            update = torch.zeros_like(visited, dtype=torch.long)
            update.scatter_add_(-1, pair_index, trigger_B.long())
            visited = visited | (update > 0)

            # in_current_route: action position only when feasible customer
            in_current_route_was = gather_by_index(in_current_route, nxt)
            in_current_route = in_current_route.scatter(
                -1,
                nxt[..., None],
                (in_current_route_was | (feasible & ~is_depot))[..., None],
            )
            # Depot return ends the current route — clear all entries.
            in_current_route = in_current_route & ~is_depot[..., None]

        # Final return-to-depot leg.
        final_loc = gather_by_index(locs, curr_node)
        total_cost = total_cost + get_distance(final_loc, depot_loc)

        # End-of-episode safety net: pairs where neither node was served on main.
        pickup_unserved = is_pickup_t & ~visited
        cost_C = (per_pickup_recourse_cost * pickup_unserved.to(float_dtype)).sum(-1)
        n_C = pickup_unserved.to(float_dtype).sum(-1)
        total_cost = total_cost + cost_C
        recourse_cost = recourse_cost + cost_C
        recourse_count = recourse_count + n_C

        if return_violation_stats:
            violation_stats = {
                "capacity_violation_count": capacity_violation_count,
                "time_window_violation_count": time_window_violation_count,
                "precedence_violation_count": precedence_violation_count,
                "open_pickup_violation_count": open_pickup_violation_count,
                "capacity_violation_rate": capacity_violation_count / recourse_count.clamp_min(1.0),
                "time_window_violation_rate": time_window_violation_count / recourse_count.clamp_min(1.0),
                "precedence_violation_rate": precedence_violation_count / recourse_count.clamp_min(1.0),
                "open_pickup_violation_rate": open_pickup_violation_count / recourse_count.clamp_min(1.0),
            }
            return total_cost, recourse_cost, recourse_count, customer_actions, violation_stats

        return total_cost, recourse_cost, recourse_count, customer_actions

    # ------------------------------------------------------------------ specs

    def _make_spec(self, generator: PDPTWGenerator):
        n = generator.num_loc + 1
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(n, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=Unbounded(shape=(1,), dtype=torch.int64, device=self.device),
            action_mask=Unbounded(shape=(n,), dtype=torch.bool, device=self.device),
            shape=(),
        )
        self.action_spec = Bounded(
            low=0,
            high=n,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ POMO

    def select_start_nodes(self, td: TensorDict, num_starts: int) -> Tensor:
        """Return ``num_starts`` start indices drawn only from the pickup nodes,
        cycling through ``pickup_idx`` so the first action is never a delivery
        (which would always trigger Trigger A).

        Note: we keep the default ``get_num_starts`` (= ``action_mask.shape[-1]``)
        so the rl4co AM decoder's 4D multistart path stays well-formed; pickups
        are simply repeated to fill the slots.
        """
        pickup_idx = td["pickup_idx"]  # [B, n_req]
        n_req = pickup_idx.size(-1)
        bs = td.shape[0]
        rep = (
            torch.arange(num_starts, device=td.device).repeat_interleave(bs)
            % n_req
        )
        repeated = pickup_idx.repeat(num_starts, 1)  # [num_starts*bs, n_req]
        return repeated.gather(-1, rep[..., None]).squeeze(-1)

    @staticmethod
    def solve(
        instances: TensorDict,
        max_runtime: float,
        num_procs: int = 1,
        solver: str = "pdptw",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instances = process_instance(instances)

        if solver not in {"pdptw", "ortools"}:
            raise ValueError("PDPTWEnv only supports `solver='pdptw'` or `solver='ortools'`.")

        try:
            from ai4co_gnn.baselines.pdptw.ortools_pdptw import solve as ortools_solve
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "PDPTW OR-Tools solver requires optional dependency `ortools`."
            ) from error

        func = partial(ortools_solve, max_runtime=max_runtime, **kwargs)
        if num_procs > 1:
            from pebble import ProcessPool

            results = []
            with ProcessPool(max_workers=num_procs) as pool:
                future = pool.map(func, instances, timeout=max_runtime)
                iterator = future.result()
                while True:
                    try:
                        results.append(next(iterator))
                    except StopIteration:
                        break
                    except TimeoutError:
                        log.warning(
                            "A PDPTW OR-Tools task exceeded %.2fs and was terminated.",
                            max_runtime,
                        )
                        results.append(
                            (
                                torch.tensor([0], dtype=torch.long),
                                torch.tensor(float("inf"), dtype=torch.float32),
                            )
                        )
                    except Exception as error:
                        log.error("A PDPTW OR-Tools task failed: %s", error)
                        results.append(
                            (
                                torch.tensor([0], dtype=torch.long),
                                torch.tensor(float("inf"), dtype=torch.float32),
                            )
                        )
        else:
            results = [func(instance) for instance in instances]

        actions, costs = zip(*results)
        max_len = max(a.shape[0] for a in actions)
        padded_actions = []
        for action in actions:
            if action.shape[0] < max_len:
                pad = torch.zeros((max_len - action.shape[0],), dtype=action.dtype)
                action = torch.cat([action, pad], dim=0)
            padded_actions.append(action)

        return torch.stack(padded_actions, dim=0).long(), torch.stack(list(costs), dim=0)
