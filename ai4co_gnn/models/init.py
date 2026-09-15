from math import pi as PI

import torch


class MTVRPInitEmbedding(torch.nn.Module):
    """
    MTVRP Initial Embedding Layer.
    It embeds the node and depot features into a higher-dimensional space.
    The node features include coordinates, demands, time windows, and service times,
    while the depot features include the open state, the depot's coordinates, distance limit, and depot close time window.
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        node_dim: int = 7,
        global_dim: int = 3 + 2,
    ):
        # node_dim = 7: x, y, demand_linehaul, demand_backhaul, tw start, tw end, service time
        # global_dim = 5: open_route, depot x, depot y, distance_limit, tw end depot

        super(MTVRPInitEmbedding, self).__init__()
        self.init_embed_nodes = torch.nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = torch.nn.Linear(
            global_dim, embed_dim, linear_bias
        )  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        demand_linehaul, demand_backhaul = (
            td["demand_linehaul"][..., 1:],
            td["demand_backhaul"][..., 1:],
        )
        service_time = td["service_time"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # [!] convert [0, inf] -> [0, 0] if a problem does not include the time window constraint, do not modify in-place
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)

        global_features = torch.cat(
            [
                td["open_route"].float()[..., None],  # FIXME: should we keep this?
                depot,
                td["distance_limit"][
                    ..., None
                ],  # FIXME: should we keep this? since both are in the context
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )
        global_features = torch.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        # embeddings
        depot_embedding = self.init_embed_depot(global_features)
        node_embeddings = self.init_embed_nodes(
            torch.cat(
                (
                    cities,
                    demand_linehaul[..., None],
                    demand_backhaul[..., None],
                    time_windows,
                    service_time[..., None],
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class PDPTWInitEmbedding(torch.nn.Module):
    """PDPTW Initial Embedding Layer (additive over MTVRP layout).

    The first 7 node columns and 5 global columns mirror MTVRP exactly so the
    encoder weights can be transferred verbatim. PDPTW-specific channels
    (``pickup_demand``, ``delivery_demand``, ``is_pickup``, ``is_delivery`` for
    nodes; ``vehicle_capacity``, ``speed`` for globals) are appended at the end
    and start from a fresh zero init when transferring.

    Node features (``node_dim=11``):
        [0..1]  x, y                              (MTVRP)
        [2..3]  demand_linehaul, demand_backhaul  (MTVRP — always 0 in PDPTW;
                                                   weights inherited but dormant)
        [4..6]  tw_start, tw_end, service_time    (MTVRP)
        [7..8]  pickup_demand, delivery_demand    (PDPTW-new)
        [9..10] is_pickup, is_delivery            (PDPTW-new)

    Global features (``global_dim=7``):
        [0]   open_route                          (MTVRP — always 0 in PDPTW)
        [1..2] depot_x, depot_y                   (MTVRP)
        [3]   distance_limit                      (MTVRP — inf in PDPTW, masked
                                                   to 0 by ``nan_to_num``)
        [4]   depot_tw_end                        (MTVRP)
        [5..6] vehicle_capacity, speed            (PDPTW-new)
    """

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        node_dim: int = 11,
        global_dim: int = 7,
    ):
        super().__init__()
        self.init_embed_nodes = torch.nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = torch.nn.Linear(global_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        # MTVRP-shape demand columns: emit zeros when absent (PDPTW data).
        zero_like_p = torch.zeros_like(td["pickup_demand"][..., 1:, None])
        demand_linehaul = td.get("demand_linehaul", None)
        demand_backhaul = td.get("demand_backhaul", None)
        demand_linehaul = (
            demand_linehaul[..., 1:, None] if demand_linehaul is not None else zero_like_p
        )
        demand_backhaul = (
            demand_backhaul[..., 1:, None] if demand_backhaul is not None else zero_like_p
        )
        pickup_demand = td["pickup_demand"][..., 1:, None]
        delivery_demand = td["delivery_demand"][..., 1:, None]
        service_time = td["service_time"][..., 1:, None]
        time_windows = td["time_windows"][..., 1:, :]
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)
        is_pickup = td["is_pickup"][..., 1:, None].to(cities.dtype)
        is_delivery = td["is_delivery"][..., 1:, None].to(cities.dtype)

        open_route = td.get("open_route", None)
        if open_route is None:
            open_route = torch.zeros_like(td["vehicle_capacity"], dtype=cities.dtype)
        else:
            open_route = open_route.to(cities.dtype)
        distance_limit = td.get("distance_limit", None)
        if distance_limit is None:
            distance_limit = torch.zeros_like(td["vehicle_capacity"])

        global_features = torch.cat(
            [
                open_route,
                depot.squeeze(-2),
                distance_limit,
                td["time_windows"][:, 0, 1:2],
                td["vehicle_capacity"],
                td["speed"],
            ],
            -1,
        )
        global_features = torch.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        depot_embedding = self.init_embed_depot(global_features).unsqueeze(-2)
        node_embeddings = self.init_embed_nodes(
            torch.cat(
                (
                    cities,
                    demand_linehaul,
                    demand_backhaul,
                    time_windows,
                    service_time,
                    pickup_demand,
                    delivery_demand,
                    is_pickup,
                    is_delivery,
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class PDPTWPolarInitEmbedding(torch.nn.Module):
    """PDPTW Initial Embedding Layer with polar coordinates (additive backbone).

    Mirror of :class:`PDPTWInitEmbedding` with polar columns inserted in the
    same positions as :class:`MTVRPPolarInitEmbedding`.

    Node features (``node_dim=15``):
        [0..1]   x, y                                (MTVRP)
        [2..5]   rho, theta, cos_theta, sin_theta    (MTVRP polar)
        [6..7]   demand_linehaul, demand_backhaul    (MTVRP — 0 in PDPTW)
        [8..10]  tw_start, tw_end, service_time      (MTVRP)
        [11..12] pickup_demand, delivery_demand      (PDPTW-new)
        [13..14] is_pickup, is_delivery              (PDPTW-new)
    Global features (``global_dim=7``): same layout as :class:`PDPTWInitEmbedding`.
    """

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        node_dim: int = 15,
        global_dim: int = 7,
    ):
        super().__init__()
        self.init_embed_nodes = torch.nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = torch.nn.Linear(global_dim, embed_dim, linear_bias)

    def forward(self, td):
        locs = td["locs"]
        depot, cities = locs[:, :1, :], locs[:, 1:, :]

        cartesian = locs - locs[..., 0:1, :]
        x, y = cartesian[..., 0], cartesian[..., 1]
        rho = torch.norm(cartesian, dim=-1)
        theta = torch.atan2(y, x)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)
        rho = rho / rho.max().clamp_min(1e-9)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        theta = theta / (2 * PI)
        polar_locs = torch.stack((rho, theta, cos_theta, sin_theta), dim=-1)
        td["polar_locs"] = polar_locs
        polar_locs = polar_locs[:, 1:, :]

        zero_like_p = torch.zeros_like(td["pickup_demand"][..., 1:, None])
        demand_linehaul = td.get("demand_linehaul", None)
        demand_backhaul = td.get("demand_backhaul", None)
        demand_linehaul = (
            demand_linehaul[..., 1:, None] if demand_linehaul is not None else zero_like_p
        )
        demand_backhaul = (
            demand_backhaul[..., 1:, None] if demand_backhaul is not None else zero_like_p
        )
        pickup_demand = td["pickup_demand"][..., 1:, None]
        delivery_demand = td["delivery_demand"][..., 1:, None]
        service_time = td["service_time"][..., 1:, None]
        time_windows = td["time_windows"][..., 1:, :]
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)
        is_pickup = td["is_pickup"][..., 1:, None].to(cities.dtype)
        is_delivery = td["is_delivery"][..., 1:, None].to(cities.dtype)

        open_route = td.get("open_route", None)
        if open_route is None:
            open_route = torch.zeros_like(td["vehicle_capacity"], dtype=cities.dtype)
        else:
            open_route = open_route.to(cities.dtype)
        distance_limit = td.get("distance_limit", None)
        if distance_limit is None:
            distance_limit = torch.zeros_like(td["vehicle_capacity"])

        global_features = torch.cat(
            [
                open_route,
                depot.squeeze(-2),
                distance_limit,
                td["time_windows"][:, 0, 1:2],
                td["vehicle_capacity"],
                td["speed"],
            ],
            -1,
        )
        global_features = torch.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        depot_embedding = self.init_embed_depot(global_features).unsqueeze(-2)
        node_embeddings = self.init_embed_nodes(
            torch.cat(
                (
                    cities,
                    polar_locs,
                    demand_linehaul,
                    demand_backhaul,
                    time_windows,
                    service_time,
                    pickup_demand,
                    delivery_demand,
                    is_pickup,
                    is_delivery,
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class MTVRPPolarInitEmbedding(torch.nn.Module):
    """
    MTVRP Polar Initial Embedding Layer.
    It embeds the node and depot features into a higher-dimensional space.
    Like :class:`MTVRPInitEmbedding`, but also includes polar coordinates (rho, theta) for nodes.
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        node_dim: int = 7
        + 2
        + 2,  # +2 for polar coordinates (rho, theta) +2 for cos, sin
        global_dim: int = 3 + 2,
    ):
        # node_dim = 9: x, y, rho, theta, demand_linehaul, demand_backhaul, tw start, tw end, service time
        # global_dim = 5: open_route, depot x, depot y, distance_limit, tw end depot

        super(MTVRPPolarInitEmbedding, self).__init__()
        self.init_embed_nodes = torch.nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = torch.nn.Linear(
            global_dim, embed_dim, linear_bias
        )  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        demand_linehaul, demand_backhaul = (
            td["demand_linehaul"][..., 1:],
            td["demand_backhaul"][..., 1:],
        )
        locs = td["locs"]

        cartesian = locs - locs[..., 0:1, :]
        x, y = cartesian[..., 0], cartesian[..., 1]
        rho = torch.norm(cartesian, dim=-1)
        theta = torch.atan2(y, x)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)
        rho = rho / rho.max()
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        theta = theta / (2 * PI)

        polar_locs = torch.stack((rho, theta, cos_theta, sin_theta), dim=-1)

        td["polar_locs"] = polar_locs
        polar_locs = polar_locs[:, 1:, :]  # remove depot polar locs
        service_time = td["service_time"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # [!] convert [0, inf] -> [0, 0] if a problem does not include the time window constraint, do not modify in-place
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)

        global_features = torch.cat(
            [
                td["open_route"].float()[..., None],
                depot,
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

        global_features = torch.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        # embeddings
        depot_embedding = self.init_embed_depot(global_features)
        node_embeddings = self.init_embed_nodes(
            torch.cat(
                (
                    cities,
                    polar_locs,
                    demand_linehaul[..., None],
                    demand_backhaul[..., None],
                    time_windows,
                    service_time[..., None],
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)

class LBtoPDPTWPolarInitEmbedding(torch.nn.Module):
    """PDPTW Initial Embedding Layer with polar coordinates (additive backbone).

    Mirror of :class:`PDPTWInitEmbedding` with polar columns inserted in the
    same positions as :class:`MTVRPPolarInitEmbedding`.

    Node features (``node_dim=15``):
        [0..1]   x, y                                (MTVRP)
        [2..5]   rho, theta, cos_theta, sin_theta    (MTVRP polar)
        [6..7]   demand_linehaul, demand_backhaul    (MTVRP-compatible surrogate
                                                      demands for transferred
                                                      weights)
        [8..10]  tw_start, tw_end, service_time      (MTVRP)
        [11..12] pickup_demand, delivery_demand      (PDPTW-new)
        [13..14] is_pickup, is_delivery              (PDPTW-new)
    Global features (``global_dim=7``): same layout as :class:`PDPTWInitEmbedding`.
    """

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        node_dim: int = 13,
        global_dim: int = 7,
    ):
        super().__init__()
        self.init_embed_nodes = torch.nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = torch.nn.Linear(global_dim, embed_dim, linear_bias)

    def forward(self, td):
        locs = td["locs"]
        depot, cities = locs[:, :1, :], locs[:, 1:, :]

        cartesian = locs - locs[..., 0:1, :]
        x, y = cartesian[..., 0], cartesian[..., 1]
        rho = torch.norm(cartesian, dim=-1)
        theta = torch.atan2(y, x)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)
        rho = rho / rho.max().clamp_min(1e-9)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        theta = theta / (2 * PI)
        polar_locs = torch.stack((rho, theta, cos_theta, sin_theta), dim=-1)
        td["polar_locs"] = polar_locs
        polar_locs = polar_locs[:, 1:, :]

        demand_backhaul = td["demand_backhaul"][..., 1:, None]
        demand_linehaul = td["demand_linehaul"][..., 1:, None]
        service_time = td["service_time"][..., 1:, None]
        time_windows = td["time_windows"][..., 1:, :]
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)
        is_pickup = td["is_pickup"][..., 1:, None].to(cities.dtype)
        is_delivery = td["is_delivery"][..., 1:, None].to(cities.dtype)

        open_route = td.get("open_route", None)
        if open_route is None:
            open_route = torch.zeros_like(td["vehicle_capacity"], dtype=cities.dtype)
        else:
            open_route = open_route.to(cities.dtype)
        distance_limit = td.get("distance_limit", None)
        if distance_limit is None:
            distance_limit = torch.zeros_like(td["vehicle_capacity"])

        global_features = torch.cat(
            [
                open_route,
                depot.squeeze(-2),
                distance_limit,
                td["time_windows"][:, 0, 1:2],
                td["vehicle_capacity"],
                td["speed"],
            ],
            -1,
        )
        global_features = torch.nan_to_num(
            global_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        depot_embedding = self.init_embed_depot(global_features).unsqueeze(-2)
        node_embeddings = self.init_embed_nodes(
            torch.cat(
                (
                    cities,
                    polar_locs,
                    demand_linehaul,
                    demand_backhaul,
                    time_windows,
                    service_time,
                    is_pickup,
                    is_delivery,
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)
