from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_vehicle_capacity(num_loc: int) -> int:
    """Use the same capacity heuristic as MTVRP for consistency."""
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap

class PDPTWGenerator(Generator):

    """Generator for PDPTW instances.
    The genrated insatnce contains:
    - node 0: depot
    - nodes [1, 2,..,n_req]: pickup nodes
    - nodes [n_req+1, n_req+2,..., 2*n_req]: delivery nodes

    Pairing is positional: pickup i is paired with delivery i + n_req
    """

    def __init__(
            self,
            num_loc: int = 20,
            min_loc: float = 0.0,
            max_loc: float = 1.0,
            loc_distribution: int | float | str | type | Callable = Uniform,
            capacity: float = None,
            min_demand: int = 1,
            max_demand: int = 10,
            scale_demand: bool = True,
            # Time horizon was 4.6 (rl4co MTVRP default). Too tight for PDPTW:
            # a single pair already eats svc_p + travel_pd + svc_d + travel_d0
            # + tw_width_p + tw_width_d ≈ 1.5–2.5, leaving little room for a
            # second route. Bumping to 8.0 lets pickups span a wider phase
            # range and supports 2–3 routes per episode.
            max_time: float = 8.0,
            min_service_time: float = 0.10,
            max_service_time: float = 0.20,
            # Tighter TW widths so windows are meaningful constraints;
            # combined with the wider time horizon they spread across the
            # interval rather than clustering at the start.
            min_tw_width: float = 0.15,
            max_tw_width: float = 0.50,
            speed: float = 1.0,
            variant_preset: str = "pdptw",
            max_resample_retries: int = 20,
            **kwargs,

    ):
        if num_loc % 2 != 0:
            log.warning(
                "num_loc should be even for PDPTW (half pickup, half delivery). Rounding down to nearest even number."
            )
            num_loc = num_loc + 1

        self.num_loc = num_loc
        self.num_req = num_loc // 2
        self.min_loc = min_loc
        self.max_loc = max_loc

        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = float(capacity)

        self.min_demand = min_demand
        self.max_demand = max_demand
        self.scale_demand = scale_demand

        self.max_time = float(max_time)
        self.min_service_time = float(min_service_time)
        self.max_service_time = float(max_service_time)
        self.min_tw_width = float(min_tw_width)
        self.max_tw_width = float(max_tw_width)
        self.speed = float(speed)
        self.variant_preset = variant_preset
        self.max_resample_retries = int(max_resample_retries)

        assert self.min_service_time >= 0.0
        assert self.max_service_time >= self.min_service_time
        assert self.min_tw_width > 0.0
        assert self.max_tw_width >= self.min_tw_width
        assert self.speed > 0.0

    def available_variants(self) -> list[str]:
        return ["pdptw"]

    def _sample_feasible_locs_and_widths(
        self, batch_size: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample locations and TW widths so that every request has positive slack.

        Slack per request:
            S = max_time - (travel_0p + svc_pickup + travel_pd + svc_delivery + travel_d0
                            + tw_width_pickup + tw_width_delivery)
        Rejection-resamples per-request rows where S < 0, up to ``max_resample_retries``.
        """
        locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2)).to(
            torch.float32
        )
        tw_width_pickup = torch.empty(*batch_size, self.num_req).uniform_(
            self.min_tw_width, self.max_tw_width
        )
        tw_width_delivery = torch.empty(*batch_size, self.num_req).uniform_(
            self.min_tw_width, self.max_tw_width
        )

        for _ in range(self.max_resample_retries):
            depot = locs[..., :1, :]
            pickup_locs = locs[..., 1 : self.num_req + 1, :]
            delivery_locs = locs[..., self.num_req + 1 :, :]
            travel_0p = torch.norm(depot - pickup_locs, dim=-1) / self.speed
            travel_pd = torch.norm(pickup_locs - delivery_locs, dim=-1) / self.speed
            travel_d0 = torch.norm(depot - delivery_locs, dim=-1) / self.speed
            # Use upper-bound service times so the slack check is conservative.
            slack = (
                self.max_time
                - travel_0p
                - travel_pd
                - travel_d0
                - 2.0 * self.max_service_time
                - tw_width_pickup
                - tw_width_delivery
            )
            bad = slack < 0  # [..., num_req]
            if not bool(bad.any()):
                return locs, tw_width_pickup, tw_width_delivery

            # Resample only the failing pickup/delivery pair locations and tw widths.
            new_pickup = self.loc_sampler.sample((*batch_size, self.num_req, 2)).to(
                torch.float32
            )
            new_delivery = self.loc_sampler.sample((*batch_size, self.num_req, 2)).to(
                torch.float32
            )
            new_pickup_w = torch.empty(*batch_size, self.num_req).uniform_(
                self.min_tw_width, self.max_tw_width
            )
            new_delivery_w = torch.empty(*batch_size, self.num_req).uniform_(
                self.min_tw_width, self.max_tw_width
            )
            replace = bad.unsqueeze(-1)  # [..., num_req, 1]
            pickup_locs = torch.where(replace, new_pickup, pickup_locs)
            delivery_locs = torch.where(replace, new_delivery, delivery_locs)
            tw_width_pickup = torch.where(bad, new_pickup_w, tw_width_pickup)
            tw_width_delivery = torch.where(bad, new_delivery_w, tw_width_delivery)
            locs = torch.cat((depot, pickup_locs, delivery_locs), dim=-2)

        raise RuntimeError(
            f"PDPTWGenerator failed to sample feasible TW windows after "
            f"{self.max_resample_retries} retries. Loosen max_tw_width or increase max_time."
        )

    def _generate(self, batch_size: int) -> TensorDict:

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Service times sampled first so we can clamp tw widths against them in the slack check.
        service_pickup = torch.empty(*batch_size, self.num_req).uniform_(
            self.min_service_time, self.max_service_time
        )
        service_delivery = torch.empty(*batch_size, self.num_req).uniform_(
            self.min_service_time, self.max_service_time
        )

        locs, tw_width_pickup, tw_width_delivery = self._sample_feasible_locs_and_widths(
            batch_size
        )
        depot = locs[..., :1, :]
        pickup_locs = locs[..., 1 : self.num_req + 1, :]
        delivery_locs = locs[..., self.num_req + 1 :, :]

        request_demands = (
            torch.FloatTensor(*batch_size, self.num_req)
            .uniform_(self.min_demand, self.max_demand)
            .int()
            + 1
        ).to(torch.float32)

        demand_backhaul = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)
        demand_linehaul = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)

        pickup_demand = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)
        delivery_demand = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)

        pickup_demand[..., 1 : self.num_req + 1] = request_demands
        delivery_demand[..., self.num_req + 1 :] = request_demands

        demand_backhaul[..., 1 : self.num_req + 1] = request_demands
        demand_linehaul[..., self.num_req + 1 :] = request_demands

        service_time = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)
        time_windows = torch.zeros(*batch_size, self.num_loc + 1, 2, dtype=torch.float32)
        speed = torch.full((*batch_size, 1), self.speed, dtype=torch.float32)

        service_time[..., 1 : self.num_req + 1] = service_pickup
        service_time[..., self.num_req + 1 :] = service_delivery

        time_windows[..., 0, 0] = 0.0
        time_windows[..., 0, 1] = self.max_time

        d_0p = torch.norm(depot - pickup_locs, dim=-1)
        d_pd = torch.norm(pickup_locs - delivery_locs, dim=-1)
        d_d0 = torch.norm(depot - delivery_locs, dim=-1)

        travel_0p = d_0p / self.speed
        travel_pd = d_pd / self.speed
        travel_d0 = d_d0 / self.speed

        # Latest moment the vehicle can *start* serving the pickup so that the delivery
        # window still fits inside the depot's time window.
        latest_pickup_start = (
            self.max_time
            - service_pickup
            - travel_pd
            - service_delivery
            - travel_d0
            - tw_width_delivery
        )

        pickup_start_lb = travel_0p
        # `_sample_feasible_locs_and_widths` guarantees latest_pickup_start >= pickup_start_lb,
        # so clamping is a no-op safety net.
        pickup_start_ub = torch.maximum(latest_pickup_start, pickup_start_lb)
        pickup_start = pickup_start_lb + torch.rand_like(pickup_start_lb) * (
            pickup_start_ub - pickup_start_lb
        )
        pickup_end = pickup_start + tw_width_pickup

        earliest_delivery_arrival = pickup_start + service_pickup + travel_pd
        latest_delivery_start = (
            self.max_time - service_delivery - travel_d0 - tw_width_delivery
        )
        delivery_start_lb = earliest_delivery_arrival
        delivery_start_ub = torch.maximum(latest_delivery_start, delivery_start_lb)
        delivery_start = delivery_start_lb + torch.rand_like(delivery_start_lb) * (
            delivery_start_ub - delivery_start_lb
        )
        delivery_end = delivery_start + tw_width_delivery

        time_windows[..., 1 : self.num_req + 1, 0] = pickup_start
        time_windows[..., 1 : self.num_req + 1, 1] = pickup_end
        time_windows[..., self.num_req + 1 :, 0] = delivery_start
        time_windows[..., self.num_req + 1 :, 1] = delivery_end

        # Feasibility assertions
        assert (pickup_end >= pickup_start).all()
        assert (delivery_end >= delivery_start).all()
        assert (delivery_start >= pickup_start + service_pickup + travel_pd - 1e-5).all()
        assert (
            delivery_end + service_delivery + travel_d0 <= self.max_time + 1e-5
        ).all()
        assert (pickup_start >= travel_0p - 1e-5).all()

        vehicle_capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)
        capacity_original = vehicle_capacity.clone()

        if self.scale_demand:
            demand_linehaul = demand_linehaul / vehicle_capacity
            demand_backhaul = demand_backhaul / vehicle_capacity
            vehicle_capacity = torch.ones_like(vehicle_capacity)

        pair_index = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.long)
        pickup_idx = torch.arange(1, self.num_req + 1, dtype=torch.long)
        delivery_idx = torch.arange(self.num_req + 1, self.num_loc + 1, dtype=torch.long)

        pair_index[..., 1: self.num_req + 1] = delivery_idx
        pair_index[..., self.num_req + 1:] = pickup_idx

        is_pickup = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.bool)
        is_delivery = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.bool)
        is_pickup[..., 1: self.num_req + 1] = True
        is_delivery[..., self.num_req + 1:] = True
        index_shape = [1] * len(batch_size) + [self.num_req]
        pickup_idx_batch = pickup_idx.view(*index_shape).expand(*batch_size, -1)
        delivery_idx_batch = delivery_idx.view(*index_shape).expand(*batch_size, -1)

        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "service_time": service_time,
                "time_windows": time_windows,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "distance_limit": torch.full(
                    (*batch_size, 1), float("inf"), dtype=torch.float32
                ),
                "open_route": torch.zeros((*batch_size, 1), dtype=torch.bool),
                "speed": speed,
                "pickup_idx": pickup_idx_batch,
                "delivery_idx": delivery_idx_batch,
                "pair_index": pair_index,
                "is_pickup": is_pickup,
                "is_delivery": is_delivery,
            },
            batch_size=batch_size,
        )

        return td

if __name__ == "__main__":
    generator = PDPTWGenerator(num_loc=20)

    td = generator._generate(batch_size=1)
    #print as format table to see the instance geenrated
    import pandas as pd
    num_loc = td["locs"].shape[1]
    for i in range(td.batch_size[0]):
        print(f"Instance {i}:")
        data = {
            "loc_x": td["locs"][i, :, 0].tolist(),
            "loc_y": td["locs"][i, :, 1].tolist(),
            "demand_backhaul": td["demand_backhaul"][i].tolist(),
            "demand_linehaul": td["demand_linehaul"][i].tolist(),
            "service_time": td["service_time"][i].tolist(),
            "tw_start": td["time_windows"][i, :, 0].tolist(),
            "tw_end": td["time_windows"][i, :, 1].tolist(),
        }
        df = pd.DataFrame(data)
        print(df)

















