from __future__ import annotations

import torch
import torch.nn as nn

from rl4co.models.nn.mlp import MLP
from rl4co.utils.ops import gather_by_index
from tensordict import TensorDict
from torch import Tensor


def _match_batch(tensor: Tensor, batch_size: int) -> Tensor:
    """Repeat a static batch tensor to match multistart decoder batches."""
    if tensor.shape[0] == batch_size:
        return tensor
    if batch_size % tensor.shape[0] != 0:
        raise ValueError(
            f"Cannot match tensor batch {tensor.shape[0]} to target batch {batch_size}."
        )
    repeats = batch_size // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))


class BottleneckAdapter(nn.Module):
    """Small residual MLP adapter with zero-initialized output projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adapter_dim: int = 32,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_neurons=[adapter_dim],
            hidden_act="ReLU",
        )
        last_linear = next(
            layer
            for layer in reversed(list(self.net.modules()))
            if isinstance(layer, nn.Linear)
        )
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PDPairAdapter(nn.Module):
    """Pickup-delivery residual adapter for encoded node embeddings.

    The pretrained MTVRP encoder already models geometry, capacity and time
    windows. This adapter adds request-pair information by exchanging features
    between each pickup and its paired delivery.
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_dim: int = 32,
        scalar_dim: int = 6,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scalar_projection = nn.Linear(scalar_dim, embed_dim)
        self.adapter = BottleneckAdapter(
            input_dim=4 * embed_dim,
            output_dim=embed_dim,
            adapter_dim=adapter_dim,
        )
        self.scale = scale

    def forward(self, h: Tensor, td: TensorDict) -> Tensor:
        if "pair_index" not in td.keys():
            return h

        pair_index = td["pair_index"].to(h.device)
        pair_h = gather_by_index(h, pair_index, dim=1, squeeze=False)
        scalar_h = self.scalar_projection(self._pair_scalars(td, h.dtype, h.device))
        delta = self.adapter(torch.cat([h, pair_h, pair_h - h, scalar_h], dim=-1)) #TODO test without h_pair-h
        # ancien E: h- pair_h, NV E: h, delta: pair_h
        return h + self.scale * delta

    @staticmethod
    def _pair_scalars(td: TensorDict, dtype: torch.dtype, device: torch.device) -> Tensor:
        pair_index = td["pair_index"].to(device)
        locs = td["locs"].to(device)
        pair_locs = gather_by_index(locs, pair_index, dim=1, squeeze=False)
        pair_distance = torch.norm(locs - pair_locs, dim=-1, keepdim=True)
        time_windows = torch.nan_to_num(td["time_windows"].to(device), posinf=0.0)
        pair_tw = gather_by_index(time_windows, pair_index, dim=1, squeeze=False)


        service_time = td["service_time"].to(device).unsqueeze(-1)
        pair_service_time = gather_by_index(service_time, pair_index, dim=1, squeeze=False)
        is_pickup = td["is_pickup"].to(device).unsqueeze(-1)
        is_delivery = td["is_delivery"].to(device=device, dtype=dtype).unsqueeze(-1)
        return torch.cat(
            [
                is_pickup.to(dtype),
                is_delivery.to(dtype),
                pair_tw.to(dtype),
                pair_service_time.to(dtype),
                pair_distance.to(dtype)
            ],
            dim=-1,
        )


class PDPTWLogitAdapter(nn.Module):
    """Soft pickup-delivery/time-window correction for decoder logits.

    This is intentionally not a hard feasibility mask. The PDPTW environment is
    recourse-mode: infeasible choices remain selectable but should receive a
    learned penalty when they are likely to trigger expensive recourse.
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_dim: int = 64,
        scalar_dim: int = 7,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scalar_projection = nn.Linear(scalar_dim, embed_dim)
        self.adapter = BottleneckAdapter(
            input_dim=5 * embed_dim,
            output_dim=1,
            adapter_dim=adapter_dim,
        )
        self.scale = scale

    def forward(self, td: TensorDict, embeddings: Tensor, logits: Tensor) -> Tensor:
        batch_size = logits.shape[0]
        embeddings = _match_batch(embeddings, batch_size)

        pair_index = _match_batch(td["pair_index"].to(embeddings.device), batch_size)
        pair_h = gather_by_index(embeddings, pair_index, dim=1, squeeze=False)

        current_node = td["current_node"].to(embeddings.device)
        if current_node.ndim > 1:
            current_node = current_node.squeeze(-1)
        current_node = _match_batch(current_node, batch_size)
        curr_h = gather_by_index(embeddings, current_node, dim=1, squeeze=True)
        curr_h = curr_h.unsqueeze(1).expand_as(embeddings)

        scalar_h = self.scalar_projection(
            self._candidate_scalars(td, embeddings.dtype, embeddings.device, batch_size)
        )
        delta = self.adapter(
            torch.cat(
                [embeddings, curr_h, pair_h, pair_h - embeddings, scalar_h],
                dim=-1,
            )
        ).squeeze(-1)
        return logits + self.scale * delta

    @staticmethod
    def _candidate_scalars(
        td: TensorDict,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> Tensor:
        locs = _match_batch(td["locs"].to(device), batch_size)
        pair_index = _match_batch(td["pair_index"].to(device), batch_size)
        pair_locs = gather_by_index(locs, pair_index, dim=1, squeeze=False)

        d_pair = torch.norm(locs - pair_locs, dim=-1, keepdim=True)

        time_windows = torch.nan_to_num(
            _match_batch(td["time_windows"].to(device), batch_size), posinf=0.0
        )
        pair_tw = gather_by_index(time_windows, pair_index, dim=1, squeeze=False)
        visited = _match_batch(td["visited"].to(device), batch_size).unsqueeze(-1)
        pair_visited = gather_by_index(visited.float(), pair_index, dim=1, squeeze=False)
        is_pickup = _match_batch(td["is_pickup"].to(device), batch_size).unsqueeze(-1)
        is_delivery = _match_batch(td["is_delivery"].to(device), batch_size).unsqueeze(-1)


        delivery_before_pickup = is_delivery.float() * (1.0 - pair_visited)

        return torch.cat(
            [
                is_pickup.to(dtype),
                is_delivery.to(dtype),
                pair_visited.to(dtype),
                d_pair.to(dtype),
                pair_tw.to(dtype),
                delivery_before_pickup.to(dtype),
            ],
            dim=-1,
        )


def freeze_non_adapter_parameters(module: nn.Module) -> None:
    """Freeze all parameters except adapter modules and PDPTW-specific embeddings."""
    trainable_markers = (
        "adapter",
        "pair_adapter",
        "logit_adapter",
        "scalar_projection",
        "init_embedding",
        "context_embedding",
    )
    for name, param in module.named_parameters():
        param.requires_grad = any(marker in name for marker in trainable_markers)
