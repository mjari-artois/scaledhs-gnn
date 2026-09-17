from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from ai4co_gnn.large_scale.local_solver import (
    NeuralLocalSolver,
    local_actions_to_global,
)
from ai4co_gnn.large_scale.partition import (
    angular_partition,
    propose_adjacent_group_swaps,
)
from ai4co_gnn.large_scale.solution import (
    actions_to_routes,
    evaluate_solution,
)
from ai4co_gnn.large_scale.subproblem import build_local_tensordict


@dataclass
class AngularSolveResult:
    local_indices: Tensor
    local_td: TensorDictBase
    model_output: dict[str, Any]
    global_actions: Tensor


class AngularSubproblemCoordinator:
    """Solve angular groups and select the best adjacent-swap refinement."""

    def __init__(
        self,
        model: nn.Module,
        env: Any,
        group_size: int = 50,
        num_starts: int = 8,
        device: torch.device | str | None = None,
        demand_key: str = "demand_linehaul",
        max_vehicles: int | None = None,
        refinement_enabled: bool = True,
        max_candidates: int | None = None,
    ) -> None:
        self.group_size = group_size
        self.demand_key = demand_key
        self.max_vehicles = max_vehicles
        self.refinement_enabled = refinement_enabled
        self.max_candidates = max_candidates
        self.local_solver = NeuralLocalSolver(
            model=model,
            env=env,
            device=device,
            num_starts=num_starts,
        )

    @torch.inference_mode()
    def solve(
        self,
        global_td: TensorDictBase,
        phase: str = "test",
    ) -> AngularSolveResult:
        """Solve the initial partition and all adjacent-swap candidates."""
        initial_groups = angular_partition(
            global_td["locs"],
            group_size=self.group_size,
        )
        partitions = [initial_groups]
        if self.refinement_enabled:
            swap_candidates = propose_adjacent_group_swaps(initial_groups)
            if self.max_candidates is not None:
                if self.max_candidates < 0:
                    raise ValueError("max_candidates must be non-negative or None")
                swap_candidates = swap_candidates[: self.max_candidates]
            partitions.extend(swap_candidates)

        batch_size = global_td.batch_size[0]
        num_groups = initial_groups.size(1)
        candidate_outputs: list[dict[str, Any]] = []
        candidate_actions: list[Tensor] = []
        candidate_metrics = []

        if self.demand_key not in global_td:
            raise KeyError(
                f"Demand field {self.demand_key!r} was not found. "
                f"Available fields: {list(global_td.keys())}"
            )

        for groups in partitions:
            local_td = build_local_tensordict(global_td, groups)
            output = self.local_solver.solve(local_td, phase=phase)
            actions = output["actions"]

            expected_local_batch = batch_size * num_groups
            if actions.ndim != 2 or actions.size(0) != expected_local_batch:
                raise ValueError(
                    "The local policy must return actions shaped "
                    f"[{expected_local_batch}, route_length]; got "
                    f"{tuple(actions.shape)}"
                )

            global_actions = local_actions_to_global(
                local_actions=actions,
                local_indices=groups.to(actions.device),
            )
            routes_per_instance = actions_to_routes(
                global_actions=global_actions,
                num_instances=batch_size,
                num_groups=num_groups,
            )

            metrics_for_candidate = []
            for instance_idx, routes in enumerate(routes_per_instance):
                metrics = evaluate_solution(
                    routes=routes,
                    locs=global_td["locs"][instance_idx],
                    demand=global_td[self.demand_key][instance_idx],
                    vehicle_capacity=global_td["vehicle_capacity"][instance_idx],
                    expected_customers=global_td["locs"].size(1) - 1,
                    max_vehicles=self.max_vehicles,
                )
                metrics_for_candidate.append(metrics)

            candidate_outputs.append(output)
            candidate_actions.append(global_actions)
            candidate_metrics.append(metrics_for_candidate)

        # Feasible results rank ahead of infeasible ones; ties retain the
        # original angular partition because it is candidate zero.
        best_candidate = []
        for instance_idx in range(batch_size):
            best_idx = min(
                range(len(partitions)),
                key=lambda candidate_idx: (
                    not candidate_metrics[candidate_idx][instance_idx].feasible,
                    candidate_metrics[candidate_idx][instance_idx].cost,
                ),
            )
            best_candidate.append(best_idx)

        selected_indices = torch.stack(
            [partitions[best_candidate[b]][b] for b in range(batch_size)]
        )
        selected_actions = torch.cat(
            [
                candidate_actions[best_candidate[b]][
                    b * num_groups : (b + 1) * num_groups
                ]
                for b in range(batch_size)
            ],
            dim=0,
        )

        selected_output = _select_candidate_outputs(
            outputs=candidate_outputs,
            best_candidate=best_candidate,
            batch_size=batch_size,
            num_groups=num_groups,
        )
        selected_local_td = build_local_tensordict(global_td, selected_indices)

        return AngularSolveResult(
            local_indices=selected_indices,
            local_td=selected_local_td,
            model_output=selected_output,
            global_actions=selected_actions,
        )


def _select_candidate_outputs(
    outputs: list[dict[str, Any]],
    best_candidate: list[int],
    batch_size: int,
    num_groups: int,
) -> dict[str, Any]:
    """Select per-instance local output tensors from winning candidates."""
    selected: dict[str, Any] = {}
    local_batch_size = batch_size * num_groups

    for key, first_value in outputs[0].items():
        values = [output.get(key) for output in outputs]

        if (
            isinstance(first_value, Tensor)
            and first_value.ndim > 0
            and first_value.size(0) == local_batch_size
            and all(
                isinstance(value, Tensor)
                and value.shape == first_value.shape
                for value in values
            )
        ):
            stacked = torch.stack(values).reshape(
                len(outputs),
                batch_size,
                num_groups,
                *first_value.shape[1:],
            )
            selected[key] = torch.cat(
                [
                    stacked[best_candidate[b], b]
                    for b in range(batch_size)
                ],
                dim=0,
            )
        else:
            selected[key] = first_value

    return selected
