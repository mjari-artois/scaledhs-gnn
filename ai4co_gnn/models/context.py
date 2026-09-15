import torch

from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.models.nn.env_embeddings.context import MTVRPContext as MTVRPContextBase


class MTVRPContext(MTVRPContextBase):
    """Context embedding for Multi-Task VRPEnv.
    Project the following to the embedding space:
        - current node embedding
        - remaining_linehaul_capacity (vehicle_capacity - used_capacity_linehaul)
        - remaining_backhaul_capacity (vehicle_capacity - used_capacity_backhaul)
        - current time
        - remaining route length (instead of distance limit)
        - open route indicator
    """

    def __init__(self, embed_dim):
        super().__init__(embed_dim)

    def _state_embedding(self, embeddings, td):
        remaining_linehaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]

        remaining_route_length = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"], posinf=10.0
        )
        open_route = td["open_route"]
        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                remaining_route_length,
                open_route,
            ],
            -1,
        )


class PDPTWContext(EnvContext):
    """Context embedding for the PDPTWEnv (additive backbone over MTVRPContext).

    The first 5 state features mirror the MTVRP context layout exactly so
    pretrained ``project_context`` weights for those slots transfer verbatim.
    PDPTW-specific signals are appended at the end.

    Layout (``step_context_dim = embed + 7``):
        [emb..emb]            current node embedding (gathered upstream)
        [emb+0]   remaining_linehaul_capacity   (MTVRP — 0 in PDPTW; dormant)
        [emb+1]   remaining_backhaul_capacity   (MTVRP — 0 in PDPTW; dormant)
        [emb+2]   current_time                  (MTVRP — semantically shared)
        [emb+3]   remaining_route_length        (MTVRP — 0 in PDPTW; dormant)
        [emb+4]   open_route                    (MTVRP — 0 in PDPTW; dormant)
        [emb+5]   remaining_capacity            (PDPTW-new)
        [emb+6]   n_unserved_customers          (PDPTW-new)
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim=embed_dim, step_context_dim=embed_dim + 7)

    def _state_embedding(self, embeddings, td):
        current_time = td["current_time"]
        zero_t = torch.zeros_like(current_time)
        # MTVRP-shape slots: zero unless td also carries the MTVRP-specific keys
        # (which it doesn't for PDPTW). The encoder weights for these slots
        # transfer from MTVRP but contribute zero gradient when the inputs are 0.
        remaining_linehaul_capacity = zero_t
        remaining_backhaul_capacity = zero_t
        remaining_route_length = zero_t
        open_route = zero_t

        remaining_capacity = td["vehicle_capacity"] - td["current_load"]
        n_unserved = (
            (~td["visited"][..., 1:]).to(current_time.dtype).sum(-1, keepdim=True)
        )

        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                remaining_route_length,
                open_route,
                remaining_capacity,
                n_unserved,
            ],
            -1,
        )

class LBtoPDPTWContext(EnvContext):
    """Context embedding for the PDPTWEnv (additive backbone over MTVRPContext).

    The first 5 state features mirror the MTVRP context layout exactly so
    pretrained ``project_context`` weights for those slots transfer verbatim.
    PDPTW-specific signals are appended at the end.

    Layout (``step_context_dim = embed + 7``):
        [emb..emb]            current node embedding (gathered upstream)
        [emb+0]   remaining_linehaul_capacity   (MTVRP — 0 in PDPTW; dormant)
        [emb+1]   remaining_backhaul_capacity   (MTVRP — 0 in PDPTW; dormant)
        [emb+2]   current_time                  (MTVRP — semantically shared)
        [emb+3]   remaining_route_length        (MTVRP — 0 in PDPTW; dormant)
        [emb+4]   open_route                    (MTVRP — 0 in PDPTW; dormant)
        [emb+5]   remaining_capacity            (PDPTW-new)
        [emb+6]   n_unserved_customers          (PDPTW-new)
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim=embed_dim, step_context_dim=embed_dim + 6)

    def _state_embedding(self, embeddings, td):
        current_time = td["current_time"]
        zero_t = torch.zeros_like(current_time)
        # MTVRP-shape slots: zero unless td also carries the MTVRP-specific keys
        # (which it doesn't for PDPTW). The encoder weights for these slots
        # transfer from MTVRP but contribute zero gradient when the inputs are 0.
        remaining_linehaul_capacity = (
                td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
                td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]

        remaining_route_length = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"], posinf=10.0
        )
        open_route = zero_t

        n_unserved = (
            (~td["visited"][..., 1:]).to(current_time.dtype).sum(-1, keepdim=True)
        )

        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                remaining_route_length,
                open_route,
                n_unserved,
            ],
            -1,
        )
