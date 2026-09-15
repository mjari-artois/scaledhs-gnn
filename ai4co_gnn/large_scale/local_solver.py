from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn


class NeuralLocalSolver:
    """Run neural model on all local problems at once"""

    def __init__(
        self,
        model: nn.Module,
        env: Any,
        device: torch.device | str | None = None,
        num_starts: int = 8,
    ) -> None:
        self.device = torch.device(
            device if device is not None else self._default_device()
        )
        self.model = model.to(self.device).eval()
        self.env = env
        self.num_starts = num_starts
        self.policy = getattr(model, "policy", model)

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.inference_mode()
    def solve(
        self,
        local_td: TensorDictBase,
        phase: str = "test",
    ) -> dict[str, Any]:
        """Solve every local problem in one inference call"""
        local_td = local_td.to(self.device)
        reset_td = self.env.reset(local_td)

        output = self.policy(
            reset_td,
            self.env,
            phase=phase,
            num_starts=self.num_starts,
        )

        return output


def local_actions_to_global(
    local_actions: Tensor,
    local_indices: Tensor,
) -> Tensor:
    """Convert local node IDs returned by the model to global node IDs """
    if local_indices.ndim != 3:
        raise ValueError(
            "local_indices must have shape [B, num_groups, local_size]"
        )

    flat_mapping = local_indices.reshape(-1, local_indices.size(-1))

    if local_actions.ndim < 2:
        raise ValueError("local_actions must have at least two dimensions")

    if local_actions.size(0) != flat_mapping.size(0):
        raise ValueError(
            "The action batch must equal B * num_groups: "
            f"received {local_actions.size(0)} and {flat_mapping.size(0)}"
        )

    # Add dimensions before local_size for multi-start actions.
    mapping = flat_mapping
    while mapping.ndim < local_actions.ndim:
        mapping = mapping.unsqueeze(1)

    mapping = mapping.expand(
        *local_actions.shape[:-1],
        flat_mapping.size(-1),
    )

    return torch.gather(mapping, dim=-1, index=local_actions)
