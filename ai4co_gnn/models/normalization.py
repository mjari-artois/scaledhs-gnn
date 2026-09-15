import torch

from rl4co.utils import get_pylogger
from tensordict import TensorDict
from torch import Tensor, nn as nn
from torch_geometric.nn import LayerNorm
from torch_geometric.typing import OptTensor

log = get_pylogger(__name__)


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ScaleNorm(nn.Module):
    """From https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/x_transformer.py"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        normalizer_class = {
            "batch": nn.BatchNorm1d,
            "rms": RMSNorm,
            "scale": ScaleNorm,
            "layer": LayerNorm,
        }.get(normalization, None)
        self.normalizer = (
            normalizer_class(embed_dim, affine=True)
            if normalizer_class is not None
            else None
        )

        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )

    def forward(self, x, batch: OptTensor = None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, LayerNorm):
            if x.ndim == 3:
                self.normalizer.mode = "node"
                return self.normalizer(x)
            elif x.ndim == 2:
                self.normalizer.mode = "graph"
                return self.normalizer(x, batch)

        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


class CostNormalization:
    """
    This class normalizes costs based on the problem variant

    The normalization type can be set to:
    - "exponential": Exponential smoothing of costs
    - "cumulative": Cumulative averaging of costs
    - "no_norm": No normalization, just returns the costs as is
    - "gauss": Gaussian normalization (z-score normalization)
    The normalization operation can be set to:
    - "div": Divide costs by the mean cost of the variant
    - "sub": Subtract the mean cost of the variant from costs
    - "gauss": Apply z-score normalization (cost - mean) / std
    """

    def __init__(
        self, type="exponential", alpha: float = 0.25, epsilon: float = 1e-6
    ) -> None:
        instance_names = [
            f"{o}vrp{b}{limit}{tw}"
            for b in ["", "b"]
            for tw in ["", "tw"]
            for o in ["", "o"]
            for limit in ["", "l"]
        ]
        # Track per-variant mean, count, and sum of squared diffs (M2) for Gaussian normalization
        self.norm_vals = {
            variant: {"mean": 0.0, "count": 0, "M2": 0.0} for variant in instance_names
        }
        self.alpha = alpha
        self.epsilon = epsilon
        assert type in [
            "exponential",
            "cumulative",
            "no_norm",
            "gauss",
        ], "type must be 'exponential', 'cumulative' 'gauss', or 'no_norm'."
        self.type = type

    def __call__(
        self, td: TensorDict, costs: torch.Tensor, operation: str = "div"
    ) -> tuple[Tensor, Tensor]:
        """
        Normalize the given 'costs' based on each variant's mean cost
        using either division or subtraction.
        """
        if self.type == "no_norm":
            return costs, costs

        if self.type == "gauss":
            # Gaussian normalization requires 'gauss' operation
            operation = "gauss"

        assert operation in [
            "div",
            "sub",
            "gauss",
        ], "operation must be 'div', 'sub', or 'gauss'."

        # Make copies so we don't overwrite original values
        normalized_costs = costs.clone()
        norm_vals = torch.zeros_like(costs)

        # Temporarily store new means for each variant
        new_means = {}

        for variant in self.norm_vals.keys():
            mask = self.build_mask(variant, td)
            if not mask.any():
                continue

            # If Gaussian normalization is requested, perform z-score update & normalization:
            if operation == "gauss":
                batch_costs = costs[mask]
                batch_count = int(mask.sum().item())
                batch_mean = batch_costs.mean().item()
                batch_var = batch_costs.var(unbiased=False).item()  # population variance

                # Update running mean and M2 for this variant
                self._update_running_stats(variant, batch_mean, batch_var, batch_count)

                # Retrieve updated stats
                stats = self.norm_vals[variant]
                mean = stats["mean"]
                count = stats["count"]
                m2 = stats["M2"]
                var = m2 / max(count, 1)
                std = (var + self.epsilon) ** 0.5

                # Apply z-score normalization: (cost - mean) / std
                normalized_costs[mask] = (batch_costs - mean) / std
                norm_vals[mask] = std
            else:
                # For backward-compatible 'div' or 'sub', use original logic:
                # Compute the new average cost for this variant
                new_means[variant] = costs[mask].mean().item()

                # Update the variant using a per-variant counter (mean only)
                self.update_variant(variant, new_means[variant])

                # Apply normalization (div or sub)
                normalized_costs[mask] = self.normalize_cost(
                    variant, mask, costs, operation
                )

                # Record the final (updated) mean in norm_vals
                norm_vals[mask] = self.norm_vals[variant]["mean"]

        return normalized_costs, norm_vals

    def update_variant(self, variant: str, new_val: float):
        """
        Update the mean cost for the given variant, using exponential smoothing
        *only if* its count > 0. Otherwise, initialize directly.
        """
        vdata = self.norm_vals[variant]
        count = vdata["count"]
        old_mean = vdata["mean"]

        if count == 0:
            # First time we see this variant, set the mean = new_val
            vdata["mean"] = new_val
        else:
            if self.type == "cumulative":
                # Apply cumulative normalization
                vdata["mean"] = (old_mean * count + new_val) / (count + 1)
            elif self.type == "exponential":
                # Apply exponential smoothing
                vdata["mean"] = (1 - self.alpha) * old_mean + self.alpha * new_val
            else:
                # no normalization
                vdata["mean"] = new_val

        # Increase the update counter for this variant
        vdata["count"] += 1

    def normalize_cost(
        self, variant: str, mask: torch.Tensor, costs: torch.Tensor, operation: str
    ) -> Tensor | None:
        """
        Apply the given normalization operation for a particular variant.
        """
        mean_cost = self.norm_vals[variant]["mean"]
        if operation == "div":
            return costs[mask] / (abs(mean_cost) + self.epsilon)
        elif operation == "sub":
            return costs[mask] - mean_cost
        # 'gauss' is handled directly in __call__, so this should not be reached.

    @staticmethod
    def build_mask(variant: str, td: TensorDict) -> Tensor:
        """
        Dynamically build the boolean mask for each problem instance in 'data'
        by parsing the variant name.

        data = (node_features, global_features)
        node_features.shape = [batch_size, num_nodes, num_features_per_node]
        global_features.shape = [batch_size, num_global_features]
        """

        # Precompute relevant booleans
        backhaul = td["demand_backhaul"].sum(dim=1) > 0
        time_windows = td["time_windows"][..., 0, 1] != float("inf")  # depot end time
        open_route = td["open_route"].squeeze(dim=-1)
        distance_limit = (td["distance_limit"] != float("inf")).squeeze(dim=-1)

        # Parse the variant name to see which features are turned on

        # For each feature, if flags[feature] == True, we want the mask to be True.
        # If flags[feature] == False, we want the mask to be False => ~mask.
        def match(flag: bool, condition: torch.Tensor) -> torch.Tensor:
            return condition if flag else ~condition

        # Start with all True, refine step by step
        batch_size = td["locs"].size(0)
        mask = torch.ones(batch_size, dtype=torch.bool, device=td["locs"].device)
        mask &= match(("o" in variant), open_route)
        mask &= match(("b" in variant), backhaul)
        mask &= match(("tw" in variant), time_windows)
        mask &= match(("l" in variant), distance_limit)

        return mask

    def _update_running_stats(
        self,
        variant: str,
        batch_mean: float,
        batch_var: float,
        batch_count: int,
    ):
        """
        Merge existing (mean, M2, count) for 'variant' with a new batch of size batch_count,
        which has its own mean=batch_mean and variance=batch_var.

        Uses Welford's batch update:
          new_count = old_count + batch_count
          delta = batch_mean - old_mean
          new_mean = old_mean + delta * (batch_count / new_count)
          new_M2 = old_M2 + (batch_var * batch_count)
                   + delta^2 * (old_count * batch_count / new_count)
        """
        stats = self.norm_vals[variant]
        old_count = stats["count"]
        old_mean = stats["mean"]
        old_M2 = stats["M2"]
        k = batch_count
        new_count = old_count + k

        if old_count == 0:
            # first batch for this variant
            stats["mean"] = batch_mean
            stats["M2"] = batch_var * k
            stats["count"] = k
        else:
            # 1) compute delta between batch mean and running mean
            delta = batch_mean - old_mean

            # 2) update running mean
            stats["mean"] = old_mean + delta * (k / new_count)

            # 3) update running M2 (sum of squared deviations)
            batch_M2 = batch_var * k
            combined_M2 = old_M2 + batch_M2 + delta * delta * (old_count * k / new_count)
            stats["M2"] = combined_M2

            # 4) update count
            stats["count"] = new_count
