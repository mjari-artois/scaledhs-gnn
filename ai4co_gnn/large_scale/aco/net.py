import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class EmbNet(nn.Module):
    """Edge-message network used to embed customer-pair features."""

    def __init__(
        self,
        depth=12,
        node_features=2,
        edge_features=1,
        units=32,
        act_fn="silu",
        agg_fn="mean",
    ):
        super().__init__()
        self.depth = depth
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f"global_{agg_fn}_pool")

        self.v_lin0 = nn.Linear(node_features, units)
        self.v_lins1 = nn.ModuleList(nn.Linear(units, units) for _ in range(depth))
        self.v_lins2 = nn.ModuleList(nn.Linear(units, units) for _ in range(depth))
        self.v_lins3 = nn.ModuleList(nn.Linear(units, units) for _ in range(depth))
        self.v_lins4 = nn.ModuleList(nn.Linear(units, units) for _ in range(depth))
        self.v_bns = nn.ModuleList(gnn.BatchNorm(units) for _ in range(depth))

        self.e_lin0 = nn.Linear(edge_features, units)
        self.e_lins0 = nn.ModuleList(nn.Linear(units, units) for _ in range(depth))
        self.e_bns = nn.ModuleList(gnn.BatchNorm(units) for _ in range(depth))

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x = self.act_fn(self.v_lin0(x))
        w = self.act_fn(self.e_lin0(edge_attr))

        for i in range(self.depth):
            x0, w0 = x, w

            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)

            messages = torch.sigmoid(w0) * x2[edge_index[1]]
            aggregated = self.agg_fn(messages, edge_index[0], size=x0.size(0))

            x = x0 + self.act_fn(self.v_bns[i](x1 + aggregated))
            w = w0 + self.act_fn(
                self.e_bns[i](
                    self.e_lins0[i](w0)
                    + x3[edge_index[0]]
                    + x4[edge_index[1]]
                )
            )

        return w


class MLP(nn.Module):
    def __init__(self, units_list, act_fn="silu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList(
            nn.Linear(units_list[i], units_list[i + 1])
            for i in range(len(units_list) - 1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.lins):
            x = layer(x)
            if i < len(self.lins) - 1:
                x = self.act_fn(x)
        return torch.sigmoid(x)


class DecompositionHeuristicNet(nn.Module):
    """
    Produces a positive symmetric pairwise heuristic matrix for
    ``DecompositionACO`` from one PyG graph.

    ``pyg.x``: node features, shape [num_nodes, node_features]
    ``pyg.edge_attr``: edge features, shape [num_edges, edge_features]
    ``pyg.edge_index``: directed customer graph. For a symmetric output,
    include both directions of every customer pair.
    """

    def __init__(
        self,
        depth=12,
        node_features=2,
        edge_features=1,
        units=32,
        act_fn="silu",
        agg_fn="mean",
    ):
        super().__init__()
        self.emb_net = EmbNet(
            depth=depth,
            node_features=node_features,
            edge_features=edge_features,
            units=units,
            act_fn=act_fn,
            agg_fn=agg_fn,
        )
        self.heuristic_head = MLP([units, units, 1], act_fn=act_fn)

    def forward(self, pyg):
        edge_embeddings = self.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)
        edge_scores = self.heuristic_head(edge_embeddings).squeeze(-1)
        return self.reshape(pyg, edge_scores)

    @staticmethod
    def reshape(pyg, edge_scores):
        """Convert edge scores into the [num_nodes, num_nodes] ACO matrix."""
        n_nodes = pyg.x.size(0)
        matrix = torch.zeros(
            (n_nodes, n_nodes),
            dtype=edge_scores.dtype,
            device=edge_scores.device,
        )
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = edge_scores

        # The decomposition relation is undirected: i grouped with j is
        # equivalent to j grouped with i.
        matrix = 0.5 * (matrix + matrix.T)
        matrix.fill_diagonal_(1e-10)
        return matrix.clamp_min(1e-10)

    def freeze_gnn(self):
        for parameter in self.emb_net.parameters():
            parameter.requires_grad = False
