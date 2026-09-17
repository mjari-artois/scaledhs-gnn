from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing


class _CandidateLayer(MessagePassing):
    """Edge-aware residual message passing for candidate neighborhoods."""

    def __init__(self, units: int) -> None:
        super().__init__(aggr="mean")
        self.node_update = nn.Linear(units, units)
        self.message_update = nn.Linear(units + units, units)
        self.edge_transform = nn.Linear(units + units + units, units)
        self.node_norm = nn.LayerNorm(units)
        self.edge_norm = nn.LayerNorm(units)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        messages = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
        )

        updated_x = self.node_norm(
            x + F.silu(self.node_update(x) + messages)
        )

        updated_edges = self.edge_norm(
            edge_attr
            + F.silu(
                self.edge_transform(
                    torch.cat((edge_attr, x[source], x[target]), dim=-1)
                )
            )
        )
        return updated_x, updated_edges

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.silu(self.message_update(torch.cat((x_j, edge_attr), dim=-1)))


class GNNColonySelector(nn.Module):
    """Select colony neighborhoods with a graph neural network."""

    def __init__(
        self,
        input_features: int = 7,
        units: int = 64,
        depth: int = 3,
        graph_k: int | None = 8,
    ) -> None:
        super().__init__()
        if input_features < 1 or units < 1 or depth < 1:
            raise ValueError("input_features, units, and depth must be positive")
        self.input_features = input_features
        self.units = units
        self.graph_k = graph_k
        self.node_encoder = nn.Linear(input_features, units)
        self.edge_encoder = nn.Linear(1, units)
        self.layers = nn.ModuleList(
            [_CandidateLayer(units) for _ in range(depth)]
        )
        self.logit_head = nn.Sequential(
            nn.Linear(units * 2, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )

    @staticmethod
    def _build_graph(features: Tensor, graph_k: int | None) -> tuple[Tensor, Tensor]:
        """Build a directed candidate graph with distance-based edge weights."""
        num_candidates = features.shape[0]
        device = features.device

        if num_candidates < 2:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, 1), dtype=features.dtype, device=device),
            )

        distances = torch.cdist(features, features)
        distances.fill_diagonal_(float("inf"))
        neighbors = num_candidates - 1 if graph_k is None else min(
            max(int(graph_k), 1), num_candidates - 1
        )
        nearest = distances.topk(neighbors, largest=False, dim=1).indices
        target = torch.arange(num_candidates, device=device).repeat_interleave(neighbors)
        source = nearest.reshape(-1)
        edge_index = torch.stack((source, target), dim=0)

        # Positive bounded edge feature: nearby candidate descriptors receive
        # stronger messages.  Detach the scale so selection remains stable.
        edge_distance = distances[target, source].unsqueeze(-1)
        scale = edge_distance.detach().mean().clamp_min(1e-6)
        edge_attr = torch.exp(-edge_distance / scale)
        return edge_index, edge_attr

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Return one unnormalized selection logit per candidate."""

        if edge_index is None or edge_attr is None:
            edge_index, edge_attr = self._build_graph(features, self.graph_k)
        edge_attr = edge_attr.reshape(-1, 1).to(features)
        edge_index = edge_index.to(device=features.device, dtype=torch.long)

        x = F.silu(self.node_encoder(features))
        edges = F.silu(self.edge_encoder(edge_attr))

        for layer in self.layers:
            x, edges = layer(x, edge_index, edges)

        # Pool the candidate context, then score each candidate relative to it.
        context = x.mean(dim=0, keepdim=True).expand_as(x)
        return self.logit_head(torch.cat((x, context), dim=-1)).squeeze(-1)
