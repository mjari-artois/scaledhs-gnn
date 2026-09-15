from typing import Any, Callable, Dict, Mapping, Optional

import torch

from rl4co import utils
from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.pomo.model import POMO as POMOBase
from rl4co.utils.ops import gather_by_index, unbatchify
from tensordict import TensorDict
from torch import nn

from ai4co_gnn.data.transforms import shift_and_dihedral_augmentation
from ai4co_gnn.models.adapters import freeze_non_adapter_parameters
from ai4co_gnn.models.normalization import CostNormalization

log = utils.get_pylogger(__name__)


class POMO(POMOBase):
    """
    POMO model for RL4CO tasks with cost normalization.
    This model extends the POMO base class to include cost normalization
    and additional functionality for RL4CO-GNN environments.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8 * 8,
        augment_fn: str | Callable = shift_and_dihedral_augmentation,
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        normalization: str = "no_norm",
        transfer_from: Optional[str] = None,
        transfer_strict: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            baseline=baseline,
            num_augment=num_augment,
            augment_fn=augment_fn,
            first_aug_identity=first_aug_identity,
            feats=feats,
            num_starts=num_starts,
            **kwargs,
        )

        self.normalization = CostNormalization(type=normalization)

        if transfer_from is not None:
            self.load_from_pretrained_checkpoint(
                transfer_from, strict=transfer_strict
            )
        if freeze_backbone:
            freeze_non_adapter_parameters(self)
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            log.info(
                f"Adapter transfer freeze: trainable={trainable:,} frozen={frozen:,}"
            )

    def load_from_pretrained_checkpoint(
        self, ckpt_path: str, strict: bool = False
    ) -> dict:
        """Partial load: copy tensors from another POMO ckpt by shape alignment.

        Used for transfer learning (e.g. PDPTW initialised from an MTVRP recourse
        checkpoint). For each source tensor with a same-named target:
            - same shape          → copy verbatim;
            - same ``ndim``,
              different size      → build a zero tensor of ``tgt.shape``, copy
                                     the per-dim ``[:min(src, tgt)]`` slice
                                     into it, and load that. Channels present
                                     in the target but absent in the source
                                     start at zero (clean baseline) rather
                                     than the random init from
                                     ``super().__init__``;
            - different ``ndim``  → skip, log;
            - missing in target   → skip, log.
        """
        log.info(f"Loading transfer weights from {ckpt_path}")
        src = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_sd = src["state_dict"] if isinstance(src, dict) and "state_dict" in src else src
        src_sd = {k: v for k, v in src_sd.items() if not k.startswith("baseline.")}
        tgt_sd = self.state_dict()

        loaded, partial, mismatched, missing = [], [], [], []
        new_sd = {**tgt_sd}
        for k, v in src_sd.items():
            if k not in tgt_sd:
                missing.append(k)
                continue
            t = tgt_sd[k]
            if t.shape == v.shape:
                new_sd[k] = v
                loaded.append(k)
                continue
            if v.ndim != t.ndim:
                mismatched.append((k, tuple(v.shape), tuple(t.shape)))
                continue
            new_sd[k] = self._partial_copy(v, t)
            partial.append((k, tuple(v.shape), tuple(t.shape)))

        self.load_state_dict(new_sd, strict=strict)
        log.info(
            f"Transfer: full={len(loaded)} partial={len(partial)} "
            f"ndim-mismatch={len(mismatched)} missing={len(missing)}"
        )
        for k, src_s, tgt_s in partial:
            log.info(f"  partial-copy {k}: {src_s} -> {tgt_s}")
        for k, src_s, tgt_s in mismatched:
            log.info(f"  ndim-mismatch skip {k}: {src_s} -> {tgt_s}")
        for k in missing:
            log.info(f"  missing-in-target skip {k}")
        return {
            "loaded": loaded,
            "partial": partial,
            "mismatched": mismatched,
            "missing": missing,
        }

    @staticmethod
    def _partial_copy(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Return a tensor of ``tgt.shape`` with the per-dim ``[:min]`` slice
        copied from ``src`` and the rest set to zero. Both tensors must have
        the same number of dimensions.
        """
        assert src.ndim == tgt.ndim, "partial copy needs matching ndim"
        out = torch.zeros_like(tgt)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt.shape))
        out[slices] = src[slices].to(out.dtype)
        return out

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        costs_bks = batch.get("costs_bks")

        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(td, self.env, phase=phase, num_starts=n_start)
        if (
            getattr(self.env, "use_recourse_for_infeasible", False)
            and out.get("actions", None) is not None
            and hasattr(self.env, "get_recourse_stats")
        ):
            out.update(self.env.get_recourse_stats(td, out["actions"]))

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        reward_key = "max_aug_reward" if "max_aug_reward" in out else "reward"
        if costs_bks is not None:
            # remove -inf from costs_bks if present
            mask = costs_bks != float("-inf")
            costs_bks = costs_bks[mask].to(out[reward_key].device)
            reward = out[reward_key][mask]

            denom = costs_bks.abs()
            gap = 100.0 * (-reward - denom) / denom
            out["gap_to_bks"] = gap

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss"), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """
        Load the state dictionary, and remove the baseline for compatibility with reinforce
        """

        # remove keys starting with 'baseline.'
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("baseline.")
        }

        super().load_state_dict(state_dict, strict=strict, assign=assign)
