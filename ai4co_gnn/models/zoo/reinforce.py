from typing import Optional

import torch

from rl4co import utils
from rl4co.envs import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.pomo.model import REINFORCE as REINFORCEBase
from tensordict import TensorDict
from torch import nn

from ai4co_gnn.models.normalization import CostNormalization

log = utils.get_pylogger(__name__)


class REINFORCE(REINFORCEBase):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: REINFORCEBaseline | str = "rollout",
        baseline_kwargs: dict = {},
        reward_scale: str = None,
        normalization: str = "no_norm",
        **kwargs,
    ):
        super().__init__(
            env=env,
            policy=policy,
            baseline=baseline,
            baseline_kwargs=baseline_kwargs,
            reward_scale=reward_scale,
            **kwargs,
        )

        self.normalization = CostNormalization(type=normalization)

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        normalized_reward, norm_vals = self.normalization(td, reward)
        policy_out.update({"norm_vals": norm_vals, "norm_reward": normalized_reward})
        super().calculate_loss(
            td=td,
            batch=batch,
            policy_out=policy_out,
            reward=normalized_reward,
            log_likelihood=log_likelihood,
        )
        max_norm_reward, _ = normalized_reward.max(dim=-1)
        policy_out.update({"max_norm_reward": max_norm_reward})
        return policy_out
