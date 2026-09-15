import math

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import get_class
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.gcn import EdgeIndexFnSignature
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.ops import get_full_graph_edge_index
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch import Tensor
from torch_cluster import knn_graph
from torch_geometric.typing import Adj, OptPairTensor, Size

from ai4co_gnn.models.normalization import Normalization

try:
    from torch_geometric.nn import (
        Aggregation,
        MessagePassing,
        MultiAggregation,
        TransformerConv,
    )
except ImportError:
    TransformerConv = None

log = get_pylogger(__name__)


@torch.compiler.disable
def knn_edge_idx_fn_wrapper(td: TensorDict, num_nodes: int, k_sparse: int):
    """Wrapper for the k-NN edge index function.
    Computes the k-NN graph edge index based on the node locations in the TensorDict.
    It uses the `knn_graph` function from PyTorch Geometric torch_cluster to create a sparse graph
    """

    locs = td.get("locs")
    batch_vec = (
        torch.arange(locs.size(0), device=locs.device)
        .repeat_interleave(num_nodes)
        .to(locs.device)
    )
    if locs.is_cuda:
        edge_index = knn_graph(locs.view(-1, 2), k=k_sparse, batch=batch_vec, loop=False)
    else:
        edge_index = knn_graph(
            locs.view(-1, 2).cpu(), k=k_sparse, batch=batch_vec.cpu(), loop=False
        )

    # Connect first node of each subgraph to all other nodes and vice versa.
    # Keep all intermediate tensors on edge_index.device to avoid CPU/MPS concat errors.
    edge_device = edge_index.device
    first_node_indices = torch.arange(
        0, locs.size(0) * num_nodes, step=num_nodes, device=edge_device
    )
    all_nodes = torch.arange(locs.size(0) * num_nodes, device=edge_device)
    repeated_first_nodes = first_node_indices.repeat_interleave(num_nodes - 1)
    grouped_all_nodes = all_nodes.view(locs.size(0), num_nodes)[:, 1:].reshape(-1)
    # From first to others
    edges_fwd = torch.stack([repeated_first_nodes, grouped_all_nodes], dim=0)
    # From others to first
    edges_bwd = torch.stack([grouped_all_nodes, repeated_first_nodes], dim=0)

    edge_index = torch.cat([edge_index, edges_fwd, edges_bwd], dim=1)

    return edge_index.to(locs.device)


def full_graph_edge_idx_fn_wrapper(td: TensorDict, num_nodes: int, k_sparse: int = None):
    """Wrapper for the full graph edge index function."""
    if k_sparse is not None:
        log.warning(
            "k_sparse is set but full graph edge index is used. Ignoring k_sparse."
        )
    return get_full_graph_edge_index(num_nodes, self_loop=False).to(td.device)


@torch.compiler.disable
def pdptw_pair_augmented_knn_edge_idx_fn(
    td: TensorDict, num_nodes: int, k_sparse: int
) -> Tensor:
    """k-NN edges + extra directed edges between every pickup-delivery pair.

    Pair edges are appended at the end of the edge_index in the same global
    indexing scheme (``b * num_nodes + node``) as :func:`knn_edge_idx_fn_wrapper`,
    so downstream consumers can derive an ``is_pair_edge`` mask via
    ``td["pair_index"][batch_src, node_src] == node_dst``.
    """
    edge_index = knn_edge_idx_fn_wrapper(td, num_nodes, k_sparse)
    pair_index = td["pair_index"]  # [B, N]
    is_pickup = td["is_pickup"]  # [B, N]
    device = edge_index.device

    pu = torch.nonzero(is_pickup, as_tuple=False).to(device)  # [n_pickups, 2] (b, node)
    if pu.numel() == 0:
        return edge_index
    b_idx = pu[:, 0]
    p_node = pu[:, 1]
    d_node = pair_index.to(device)[b_idx, p_node]
    offset = b_idx * num_nodes
    src = offset + p_node
    dst = offset + d_node
    pair_edges_fwd = torch.stack([src, dst], dim=0)
    pair_edges_bwd = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, pair_edges_fwd, pair_edges_bwd], dim=1)


class UniMPEncoderBlock(nn.Module):
    """TransformerConv-based encoder block.
    from the paper, Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification
    ref : https://arxiv.org/abs/2009.03509
    """

    def __init__(
        self,
        embed_dim: int = 128,
        edge_dim: Optional[int] = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "batch",
        bias: bool = True,
        use_prenorm: bool = True,
        moe_kwargs: Optional[dict] = None,
    ):
        super(UniMPEncoderBlock, self).__init__()

        self.use_prenorm = use_prenorm

        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []

        self.gnn = TransformerConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            bias=bias,
            concat=True,
            edge_dim=edge_dim,
        )
        if moe_kwargs is not None:
            self.ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            self.ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        self.norm_gnn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        batch, num_nodes, emb_dim = x.shape
        batch_vec = (
            torch.arange(x.size(0), device=x.device)
            .repeat_interleave(num_nodes)
            .to(x.device)
        )
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745

            # (bs*num_nodes, emb_dim)
            x = x.reshape(-1, emb_dim)
            h = x + self.gnn(self.norm_gnn(x, batch=batch_vec), edge_index, edge_attr)
            h = h.reshape(batch, num_nodes, emb_dim)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            x = x.reshape(-1, emb_dim)
            h = self.norm_gnn(x + self.gnn(x, edge_index, edge_attr), batch=batch_vec)
            h = h.reshape(batch, num_nodes, emb_dim)
            h = self.norm_ffn(h + self.ffn(h))
        return h


class MPConv(MessagePassing):
    r"""
    Extension of the PyG SAGEConv to handle edge features. We add an extra
    `edge_encoder` layer that transforms edge_attr into a shape that can
    be combined with node embeddings.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: Optional[int],
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        eps: float = 1e-7,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.eps = eps
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == "lstm":
            kwargs.setdefault("aggr_kwargs", {})
            kwargs["aggr_kwargs"].setdefault("in_channels", in_channels[0])
            kwargs["aggr_kwargs"].setdefault("out_channels", in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(
                    f"'{self.__class__.__name__}' does not "
                    f"support lazy initialization with "
                    f"`project=True`"
                )
            self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = nn.Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        if isinstance(in_channels, int):
            node_dim = in_channels
        else:
            node_dim = in_channels[0]

        if edge_dim is None:
            self.edge_encoder = None
        elif edge_dim == node_dim:
            self.edge_encoder = nn.Identity()
        else:
            self.edge_encoder = nn.Linear(edge_dim, node_dim, bias=False)

        self.node_edge_lin = nn.Linear(2 * node_dim, node_dim)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        if hasattr(self.edge_encoder, "reset_parameters"):
            self.edge_encoder.reset_parameters()
        self.node_edge_lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Tensor,
        size: Size = None,
    ) -> Tensor:
        r"""
        Forward pass that additionally takes edge_attr.

        """
        if isinstance(x, Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        # If SAGEConv's `project=True`, apply the linear transform to x_src.
        if self.project and hasattr(self, "lin"):
            x_src = self.lin(x_src).relu()

        # We pass edge_attr to `propagate` as a named argument, e.g. edge_attr=edge_attr.
        # Then inside `message(...)`, we can use it via the same name
        out = self.propagate(
            edge_index,
            x=x_src,
            edge_attr=edge_attr,
            size=size,
        )

        # Then apply the `lin_l` transform from the parent class:
        out = self.lin_l(out)

        # Add the root node contribution if root_weight is True:
        x_r = x_dst
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        # Optionally normalize the output:
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if hasattr(self, "edge_encoder"):
            edge_attr = self.edge_encoder(edge_attr)

        assert x_j.size(-1) == edge_attr.size(-1)

        return x_j if edge_attr is None else x_j + edge_attr

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, "
            f'edge_encoder_dim={self.edge_encoder.in_features if hasattr(self.edge_encoder, "in_features") else 1}, )'
        )


class MPEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        edge_dim: Optional[int] = 128,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "batch",
        bias: bool = True,
        use_prenorm: bool = True,
        moe_kwargs: Optional[dict] = None,
    ):
        super(MPEncoderBlock, self).__init__()

        self.use_prenorm = use_prenorm

        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []

        self.gnn = MPConv(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            edge_dim=edge_dim,
        )

        if moe_kwargs is not None:
            self.ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            self.ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        self.norm_gnn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.gnn(self.norm_gnn(x), edge_index, edge_attr)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_gnn(x + self.gnn(x, edge_index, edge_attr))
            h = self.norm_ffn(h + self.ffn(h))
        return h


class RBFDistanceEncoding(nn.Module):
    """
    Expand normalized distances d in [0,1] into [d, d^2, RBF_K features, (optional Fourier sin/cos)].
    If fourier_feats > 0, adds sin(w_k d), cos(w_k d) with frozen random frequencies.
    """

    def __init__(self, K: int = 16, fourier_feats: int = 0, trainable: bool = False):
        super().__init__()
        if K < 1:
            raise ValueError("K (number of RBF centers) must be >=1")
        mu = torch.linspace(0.0, 1.0, K)
        if K > 1:
            delta = mu[1] - mu[0]
        else:
            delta = 1.0
        gamma = torch.full_like(mu, 1.0 / (2 * delta * delta))
        if trainable:
            self.mu = nn.Parameter(mu)
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("mu", mu)
            self.register_buffer("gamma", gamma)
        self.K = K
        self.fourier_feats = fourier_feats
        if fourier_feats > 0:
            # Frozen Gaussian frequencies
            w = torch.randn(fourier_feats) * 2 * math.pi
            self.register_buffer("w", w)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d shape: (E,) normalized to [0,1]
        d2 = d * d
        # RBF basis: exp(-gamma_k (d - mu_k)^2)
        rbf = torch.exp(-self.gamma * (d.unsqueeze(-1) - self.mu) ** 2)  # (E, K)
        feats = [d.unsqueeze(-1), d2.unsqueeze(-1), rbf]
        if self.fourier_feats > 0:
            wd = d.unsqueeze(-1) * self.w  # (E, F)
            feats.append(torch.sin(wd))
            feats.append(torch.cos(wd))
        return torch.cat(feats, dim=-1)  # (E, 2 + K + 2 * fourier_feats)


class GNNEncoder(nn.Module):
    """Gnn-based graph encoder.

    Builds a stack of :class:`TransformerConv` layers to transform environment
    nodes into latent embeddings.

    Args:
        env_name: Name of the environment for which to create node embeddings.
        embed_dim: Dimension of the embeddings.
        num_layers: Number of encoder layers.
        block: Block to use for the encoder. If ``None``, defaults to ``UniMPEncoderBlock``
        block_kwargs: Additional keyword arguments for the block.
        feedforward_hidden: Size of the feed-forward network in each layer. If
            ``None``, defaults to ``4 * embed_dim``.
        normalization: Normalization method used in each block.
        prenorm: Whether to apply normalization before the attention and FFN.
        init_embedding: Module producing the initial embeddings. If ``None``, a
            default embedding for ``env_name`` is used.
        residual: If ``True``, add the initial embeddings to the output.
        edge_idx_fn: Function to compute edge indices from the ``TensorDict``.
        dropout: Dropout probability applied after intermediate layers.
        bias: Whether to use bias in the linear layers.
        k_sparse: Number of edges to keep for each node in the graph. If ``None``, all
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int,
        num_layers: int,
        block: Optional[nn.Module] = None,
        block_kwargs: dict = None,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "batch",
        prenorm: bool = True,
        init_embedding: nn.Module = None,
        residual: bool = True,
        edge_idx_fn: EdgeIndexFnSignature = None,
        dropout: float = 0.1,
        bias: bool = True,
        sparsify: bool = True,
        k_sparse: int = 10,
        edge_features: bool = True,
        rbf_K: int = 16,
        fourier_feats: int = 1,
        trainable_rbf: bool = False,
        moe_kwargs: dict = None,
        pair_adapter: nn.Module = None,
    ):
        super().__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.residual = residual
        self.dropout = dropout
        self.pair_adapter = pair_adapter

        if init_embedding is not None:
            self.init_embedding = init_embedding
        elif self.env_name == "pdptw":
            # rl4co's env_init_embedding registry has no 'pdptw' entry; default
            # to our own implementation so the encoder is usable without an
            # explicit init_embedding override.
            from ai4co_gnn.models.init import PDPTWInitEmbedding

            self.init_embedding = PDPTWInitEmbedding(embed_dim=embed_dim)
        else:
            self.init_embedding = env_init_embedding(
                self.env_name, {"embed_dim": embed_dim}
            )

        self.block = UniMPEncoderBlock if block is None else get_class(block)
        self.k_sparse = k_sparse
        if edge_idx_fn is None:
            if sparsify:
                edge_idx_fn = knn_edge_idx_fn_wrapper
            else:
                edge_idx_fn = full_graph_edge_idx_fn_wrapper

        self.edge_idx_fn = edge_idx_fn
        self.edge_features = edge_features
        # PDPTW augments edge features with a 1-d ``is_pair_edge`` flag so the
        # GNN can distinguish topology-only pickup-delivery edges from k-NN edges.
        self._has_pair_edge_flag = env_name == "pdptw"
        if edge_features:
            self.rbf_K = rbf_K
            self.fourier_feats = fourier_feats
            self.edge_rbf = RBFDistanceEncoding(
                K=rbf_K, fourier_feats=fourier_feats, trainable=trainable_rbf
            )
            # Edge feature layout: sin(angle), cos(angle) -> 2
            # Distance expansion: d, d^2, K RBF, 2*fourier_feats (if any) -> 2 + K + 2*fourier_feats
            # Optional is_pair_edge flag for PDPTW: +1
            extra_edge_dim = 1 if self._has_pair_edge_flag else 0
            in_dim = 2 + (2 + rbf_K + 2 * fourier_feats) + extra_edge_dim
            self.edge_projection = nn.Linear(in_dim, embed_dim, bias=bias)

        # Define the GNN layers
        if block_kwargs is None:
            block_kwargs = {}
        block_kwargs.setdefault("embed_dim", embed_dim)
        block_kwargs.setdefault("normalization", normalization)
        block_kwargs.setdefault("use_prenorm", prenorm)
        block_kwargs.setdefault("feedforward_hidden", feedforward_hidden)

        if not edge_features:
            block_kwargs.setdefault("edge_dim", None)
        else:
            block_kwargs.setdefault("edge_dim", embed_dim)

        block_kwargs.setdefault(
            "moe_kwargs", moe_kwargs
        )  # Check if we can pass moe_kwargs to the block like this

        self.gnn_layers = nn.ModuleList(
            [self.block(**block_kwargs) for _ in range(num_layers)]
        )

    def forward(
        self, td: TensorDict, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)
        bs, num_nodes, emb_dim = init_h.shape
        if self.k_sparse is not None:
            edge_index = self.edge_idx_fn(td, num_nodes, self.k_sparse)
        else:
            edge_index = self.edge_idx_fn(td, num_nodes)

        if not self.edge_features:
            edge_attr = None
        else:
            src = edge_index[0]
            dst = edge_index[1]

            batch_src = src // num_nodes
            node_src = src % num_nodes
            node_dst = dst % num_nodes
            locs = td.get("locs")
            distances = torch.cdist(
                locs, locs, p=2
            )  # [batch * num_nodes, batch * num_nodes]

            raw_dist = distances[batch_src, node_src, node_dst]  # (E,)

            # Per-batch max for scale invariance
            max_d_per_batch = distances.amax(dim=(1, 2)).clamp_min(1e-6)  # (B,)
            d_norm = raw_dist / max_d_per_batch[batch_src]  # normalized distance in [0,1]

            angle = torch.atan2(
                locs[batch_src, node_dst, 1] - locs[batch_src, node_src, 1],  # y
                locs[batch_src, node_dst, 0] - locs[batch_src, node_src, 0],
            )  # shape => [E, ]
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)

            dist_feats = self.edge_rbf(d_norm)  # shape => [E, 2 + K + 2 * fourier_feats]
            edge_feature_list = [
                dist_feats,
                cos_angle.unsqueeze(-1),
                sin_angle.unsqueeze(-1),
            ]
            if self._has_pair_edge_flag:
                pair_index = td["pair_index"]
                is_pair_edge = (
                    pair_index[batch_src, node_src] == node_dst
                ).to(dist_feats.dtype)
                edge_feature_list.append(is_pair_edge.unsqueeze(-1))
            edge_features = torch.cat(edge_feature_list, dim=-1)

            edge_attr = self.edge_projection(edge_features)  # shape => [E, embed_dim]

            # Edge angles (radians)
        update_node_feature = init_h
        for layer in self.gnn_layers[:-1]:
            update_node_feature = layer(
                update_node_feature, edge_index=edge_index, edge_attr=edge_attr
            )
            if self.pair_adapter is not None:
                update_node_feature = self.pair_adapter(update_node_feature, td)
            update_node_feature = F.relu(update_node_feature)
            update_node_feature = F.dropout(
                update_node_feature, training=self.training, p=self.dropout
            )

        # last layer without relu activation and dropout
        update_node_feature = self.gnn_layers[-1](
            update_node_feature, edge_index, edge_attr=edge_attr
        )
        if self.pair_adapter is not None:
            update_node_feature = self.pair_adapter(update_node_feature, td)

        # De-batch the graph
        update_node_feature = update_node_feature.view(bs, num_nodes, emb_dim)

        # Residual
        if self.residual:
            update_node_feature = update_node_feature + init_h

        return update_node_feature, init_h
