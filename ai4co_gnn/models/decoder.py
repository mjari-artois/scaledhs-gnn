from typing import Tuple

import torch
import torch.nn as nn

from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from rl4co.utils import get_pylogger
from rl4co.utils.ops import gather_by_index, get_distance
from tensordict import TensorDict
from torch import Tensor

log = get_pylogger(__name__)


class AttentionAndDistanceModelDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "mtvrp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
        logit_adapter: nn.Module = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            pointer=pointer,
            moe_kwargs=moe_kwargs,
        )

        self.edge_distance_scale = nn.Parameter(torch.tensor(-1.0))
        self.W_dist = nn.Linear(1, embed_dim, bias=False)
        self.logit_adapter = logit_adapter

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):

        glimpse_k, glimpse_v, logit_k = super()._compute_kvl(cached, td)

        # Add distance term to glimpse_k and glimpse_v
        curr_node = td["current_node"]
        locs = td["locs"]
        if locs.ndim == 4:
            d_ij = get_distance(gather_by_index(locs, curr_node), locs)[
                ..., 0, :
            ]  # [B, N, 1]  # i (current) -> j (next) (if multi start take the first instance)
            d_j0 = get_distance(locs, locs[..., 0:1, :])[
                ..., 0, :
            ]  # j (next) -> 0 (depot)

        else:
            d_ij = get_distance(gather_by_index(locs, curr_node)[..., None, :], locs)
            d_j0 = get_distance(locs, locs[..., 0:1, :])

        d_ij = d_ij + (d_j0 * ~td["open_route"][..., 0, :])
        d_ij_project = self.W_dist(d_ij.unsqueeze(-1)) if self.W_dist is not None else 0.0
        glimpse_k = glimpse_k + d_ij_project
        glimpse_v = glimpse_v + d_ij_project

        return glimpse_k, glimpse_v, logit_k

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:

        logits, mask = super().forward(td=td, cached=cached, num_starts=num_starts)
        curr_node = td["current_node"]
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)

        logits = logits + (d_ij + (d_j0 * ~td["open_route"])) * self.edge_distance_scale
        if self.logit_adapter is not None:
            logits = self.logit_adapter(td, cached.node_embeddings, logits)

        return logits, mask


class ServedOnlyMaskAttentionAndDistanceModelDecoder(AttentionAndDistanceModelDecoder):
    """Backward-compatible alias for older checkpoints."""

    pass


class ServedOnlyMaskAttentionModelDecoder(AttentionModelDecoder):
    """Backward-compatible alias for older checkpoints."""

    pass
