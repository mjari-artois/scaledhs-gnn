from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from ai4co_gnn.large_scale.local_solver import NeuralLocalSolver, local_actions_to_global
from ai4co_gnn.large_scale.partition import angular_partition
from ai4co_gnn.large_scale.subproblem import build_local_tensordict

@dataclass
class AngularSolveResult:

    local_indices: Tensor
    local_td: TensorDictBase
    model_output: dict[str, Any]
    global_actions: Tensor

class AngularSubproblemCoordinator:
    """Split a large CVRP instance and solve each subproblem with a neural model"""

    def __init__(
        self,
        model: nn.Module,
        env: Any,
        group_size: int = 50,
        num_starts: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        self.group_size = group_size
        self.local_solver = NeuralLocalSolver(
            model=model,
            env=env,
            device=device,
            num_starts=num_starts
        )

    @torch.inference_mode()
    def solve(
        self,
        global_td: TensorDictBase,
        phase: str = "test",
    ) -> AngularSolveResult:
        """Split a large CVRP instance into subproblems and solve each subproblem with a neural model"""
        local_indices = angular_partition(global_td["locs"], group_size=self.group_size)
        local_td = build_local_tensordict(global_td, local_indices)
        model_output = self.local_solver.solve(local_td, phase=phase)
        local_actions = model_output["actions"]
        local_indices = local_indices.to(local_actions.device)
        global_actions = local_actions_to_global(
            local_actions=local_actions,
            local_indices=local_indices,
        )

        return AngularSolveResult(
            local_indices=local_indices,
            local_td=local_td,
            model_output=model_output,
            global_actions=global_actions,
        )