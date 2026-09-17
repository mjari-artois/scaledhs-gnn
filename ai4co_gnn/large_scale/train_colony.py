from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "train_colony.py requires lightning; install lightning>=2.0"
    ) from exc

from ai4co_gnn.large_scale.colony import ColonyCVRP


@dataclass
class ColonyEpisode:
    """a feasible solution for that instance."""

    global_td: Any
    initial_routes: Sequence[Sequence[int]]
    reference_cost: float | None = None


class ColonyEpisodeDataset(Dataset[ColonyEpisode]):
    """Dataset wrapper used with ``collate_fn=lambda items: items``."""

    def __init__(self, episodes: Sequence[ColonyEpisode]) -> None:
        self.episodes = list(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> ColonyEpisode:
        return self.episodes[index]


class ColonySelectorLightning(pl.LightningModule):
    """Train a colony selector while keeping local routing frozen."""

    def __init__(
        self,
        selector: nn.Module,
        repair_fn: Callable,
        *,
        rounds: int = 8,
        n_ants: int = 2,
        learning_rate: float = 1e-4,
        entropy_weight: float = 0.01,
        baseline_momentum: float = 0.9,
        max_local_customers: int = 50,
        candidate_pool_size: int = 32,
        evaporation: float = 0.05,
        pheromone_alpha: float = 1.0,
        exploration: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= baseline_momentum < 1.0:
            raise ValueError("baseline_momentum must be in [0, 1)")
        self.selector = selector
        self.repair_fn = repair_fn
        self.rounds = rounds
        self.n_ants = n_ants
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.baseline_momentum = baseline_momentum
        self.search_kwargs = {
            "max_local_customers": max_local_customers,
            "candidate_pool_size": candidate_pool_size,
            "n_ants": n_ants,
            "rounds": rounds,
            "evaporation": evaporation,
            "pheromone_alpha": pheromone_alpha,
            "exploration": exploration,
        }
        self.register_buffer("running_baseline", torch.zeros(()))
        self.register_buffer("baseline_initialized", torch.zeros((), dtype=torch.bool))

    def _new_search(self, episode: ColonyEpisode) -> ColonyCVRP:
        search = ColonyCVRP(
            global_td=episode.global_td,
            initial_routes=episode.initial_routes,
            repair_fn=self.repair_fn,
            selector=self.selector,
            device=self.device,
            **self.search_kwargs,
        )
        # Colony inference uses eval mode by default; training needs gradients
        # through the selector logits while the repair remains frozen.
        search.selector = self.selector
        return search

    def _run_episode(self, episode: ColonyEpisode) -> tuple[Tensor, Tensor, Tensor, float]:
        search = self._new_search(episode)
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        rewards: list[Tensor] = []
        denominator = max(search.initial_cost, 1e-8)

        for _ in range(self.rounds):
            search._evaporate()
            for _ in range(self.n_ants):
                candidates = search.propose_candidates()
                trace = search.select_candidate(candidates, return_trace=True)
                if trace is None:
                    continue
                candidate, log_prob, entropy = trace
                before = search.solution_cost(search.routes)
                accepted = search._try_candidate(candidate)
                after = search.solution_cost(search.routes)
                reward = (before - after) / denominator if accepted else 0.0
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(log_prob.new_tensor(reward))

        if not rewards:
            zero = self.selector_logit_zero()
            empty = zero.reshape(0)
            return empty, empty, empty, 0.0

        returns: list[Tensor] = [rewards[-1].new_zeros(()) for _ in rewards]
        running = rewards[-1].new_zeros(())
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + running
            returns[index] = running

        returns_tensor = torch.stack(returns)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        return (
            log_prob_tensor,
            entropy_tensor,
            returns_tensor.detach(),
            float(returns_tensor.mean().detach()),
        )

    def selector_logit_zero(self) -> Tensor:
        """Create a differentiable zero when an episode has no candidates."""
        parameter = next(self.selector.parameters(), None)
        if parameter is None:
            return torch.zeros((), device=self.device)
        return parameter.sum() * 0.0

    def training_step(self, batch: list[ColonyEpisode] | ColonyEpisode, batch_idx: int) -> Tensor:
        episodes = batch if isinstance(batch, list) else [batch]
        losses: list[Tensor] = []
        batch_returns: list[float] = []

        for episode in episodes:
            log_probs, entropies, returns, mean_return = self._run_episode(episode)
            if log_probs.numel() == 0:
                losses.append(self.selector_logit_zero())
                batch_returns.append(mean_return)
                continue
            baseline = self.running_baseline.detach()
            advantages = (returns - baseline).detach()
            losses.append(
                -(advantages * log_probs).mean()
                - self.entropy_weight * entropies.mean()
            )
            batch_returns.append(mean_return)

        loss = torch.stack(losses).mean()
        mean_return = sum(batch_returns) / max(len(batch_returns), 1)
        observed = torch.tensor(mean_return, device=self.device)
        if not bool(self.baseline_initialized):
            self.running_baseline.copy_(observed)
            self.baseline_initialized.fill_(True)
        else:
            self.running_baseline.mul_(self.baseline_momentum).add_(
                observed * (1.0 - self.baseline_momentum)
            )

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/return", observed, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: list[ColonyEpisode] | ColonyEpisode,
        batch_idx: int,
    ) -> dict[str, Tensor]:
        """Run inference search and report cost, improvement, and gap."""
        episodes = batch if isinstance(batch, list) else [batch]
        costs: list[float] = []
        improvements: list[float] = []
        gaps: list[float] = []
        self.selector.eval()
        with torch.no_grad():
            for episode in episodes:
                search = self._new_search(episode)
                result = search.run()
                cost = float(result["cost"])
                initial = float(result["initial_cost"])
                reference = episode.reference_cost
                costs.append(cost)
                improvements.append((initial - cost) / max(initial, 1e-8))
                gaps.append(
                    (cost - reference) / max(reference, 1e-8)
                    if reference is not None else 0.0
                )

        output = {
            "cost": torch.tensor(costs, device=self.device),
            "improvement": torch.tensor(improvements, device=self.device),
            "gap": torch.tensor(gaps, device=self.device),
        }
        self.log("val/cost", output["cost"].mean(), prog_bar=True, on_epoch=True)
        self.log("val/improvement", output["improvement"].mean(), on_epoch=True)
        self.log("val/gap", output["gap"].mean(), prog_bar=True, on_epoch=True)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.selector.parameters(), lr=self.learning_rate)


def build_colony_dataloader(
    episodes: Sequence[ColonyEpisode] | Dataset[ColonyEpisode],
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a loader that preserves each episode as an independent object."""
    dataset = episodes if isinstance(episodes, Dataset) else ColonyEpisodeDataset(episodes)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda items: items,
    )


def train_colony_selector(
    selector: nn.Module,
    repair_fn: Callable,
    episodes: Sequence[ColonyEpisode],
    *,
    validation_episodes: Sequence[ColonyEpisode] | None = None,
    max_epochs: int = 10,
    batch_size: int = 1,
    trainer_kwargs: dict[str, Any] | None = None,
    **module_kwargs: Any,
) -> ColonySelectorLightning:
    """Train and return a selector using Lightning's standard Trainer."""
    module = ColonySelectorLightning(
        selector=selector,
        repair_fn=repair_fn,
        **module_kwargs,
    )
    loader = build_colony_dataloader(episodes, batch_size=batch_size)
    validation_loader = (
        build_colony_dataloader(
            validation_episodes,
            batch_size=batch_size,
            shuffle=False,
        )
        if validation_episodes is not None
        else None
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        **(trainer_kwargs or {}),
    )
    trainer.fit(
        module,
        train_dataloaders=loader,
        val_dataloaders=validation_loader,
    )
    return module
