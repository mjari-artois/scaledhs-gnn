from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from tensordict import TensorDictBase
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from ai4co_gnn.large_scale.aco.aco import DecompositionACO
from ai4co_gnn.large_scale.data import load_uniform_npz
from ai4co_gnn.large_scale.local_solver import (
    NeuralLocalSolver,
    local_actions_to_global,
)
from ai4co_gnn.large_scale.solution import actions_to_routes, evaluate_solution
from ai4co_gnn.large_scale.subproblem import build_local_tensordict


class TensorDictDataset(Dataset):
    """A map-style wrapper that lets a TensorDict use a PyTorch DataLoader."""

    def __init__(self, data: TensorDictBase):
        self.data = data

    def __len__(self):
        return self.data.batch_size[0]

    def __getitem__(self, index):
        return self.data[index]


def _stack_tensordicts(items):
    return torch.stack(items, dim=0)


class DecompositionDataModule(pl.LightningDataModule):
    """DataModule for NPZ files loaded by ``load_uniform_npz``."""

    def __init__(
        self,
        train_source: str,
        validation_source: str,
        test_source: str,
        train_path: str | Path | None,
        validation_path: str | Path | None,
        test_path: str | Path | None,
        train_data_size: int,
        validation_data_size: int,
        test_data_size: int,
        batch_size: int,
        num_workers: int = 0,
        dataset_format: str = "uniform_npz",
        env: Any | None = None,
    ):
        super().__init__()
        self.train_source = train_source
        self.validation_source = validation_source
        self.test_source = test_source
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path
        self.train_data_size = train_data_size
        self.validation_data_size = validation_data_size
        self.test_data_size = test_data_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_format = dataset_format
        self.env = env

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_data = self._dataset(
                source=self.train_source,
                path=self.train_path,
                size=self.train_data_size,
                phase="train",
            )
            self.validation_data = self._dataset(
                source=self.validation_source,
                path=self.validation_path,
                size=self.validation_data_size,
                phase="val",
            )
            self.test_data = self._dataset(
                source=self.test_source,
                path=self.test_path,
                size=self.test_data_size,
                phase="test",
            )

    def _dataset(self, source: str, path: str | Path | None, size: int, phase: str):
        if source == "generated":
            if self.env is None:
                raise ValueError("env is required for generated datasets")
            return self.env.dataset(batch_size=[size], phase=phase, filename=None)

        if source != "file":
            raise ValueError("dataset source must be 'generated' or 'file'")
        if path is None:
            raise ValueError("A dataset path is required when source='file'")

        data = self._load(path)
        if size > 0:
            data = data[:size]
        return TensorDictDataset(data)

    def _load(self, path: str | Path):
        if self.dataset_format == "uniform_npz":
            return load_uniform_npz(path)
        if self.dataset_format == "rl4co_npz":
            if self.env is None:
                raise ValueError("env is required for dataset_format='rl4co_npz'")
            return self.env.load_data(path)
        raise ValueError(
            f"Unknown dataset_format '{self.dataset_format}'. "
            "Use 'uniform_npz' or 'rl4co_npz'."
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=getattr(self.train_data, "collate_fn", _stack_tensordicts),
        )

    def val_dataloader(self):
        validation_loader = DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=getattr(self.validation_data, "collate_fn", _stack_tensordicts),
        )
        test_loader = self._test_loader()
        return [validation_loader, test_loader]

    def test_dataloader(self):
        return self._test_loader()

    def _test_loader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=getattr(self.test_data, "collate_fn", _stack_tensordicts),
        )


class ACODecompositionTrainer(pl.LightningModule):
    """
    Train a decomposition heuristic with final merged-solution costs.

    For each large instance in a batch, the module:
      1. builds a complete customer graph and gets a heuristic from the GNN;
      2. samples decompositions using ``DecompositionACO``;
      3. solves all same-size subproblems together with ``NeuralLocalSolver``;
      4. reconstructs the complete solution and applies REINFORCE.

    The small-instance solver is frozen. Gradients flow only through the
    GNN action log-probabilities returned by the ACO constructor.
    """

    def __init__(
        self,
        heuristic_net: torch.nn.Module,
        local_solver_model: torch.nn.Module,
        env: Any,
        n_ants: int = 8,
        max_subproblem_size: int = 50,
        aco_iterations: int = 1,
        solver_num_starts: int = 1,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        close_weight: float = 0.1,
        demand_key: str = "demand_linehaul",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        infeasible_penalty: float = 1e6,
        global_improver: Any | None = None,
        test_every_n_epochs: int = 5,
    ):
        super().__init__()
        if n_ants < 2:
            raise ValueError("n_ants must be at least 2 for the REINFORCE baseline")

        self.heuristic_net = heuristic_net
        self.n_ants = n_ants
        self.max_subproblem_size = max_subproblem_size
        self.aco_iterations = aco_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.close_weight = close_weight
        self.demand_key = demand_key
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.infeasible_penalty = infeasible_penalty
        self.global_improver = global_improver
        self.test_every_n_epochs = test_every_n_epochs

        # This object owns a frozen, inference-only black-box solver.
        self.local_solver = NeuralLocalSolver(
            model=local_solver_model,
            env=env,
            num_starts=solver_num_starts,
        )
        for parameter in self.local_solver.model.parameters():
            parameter.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.heuristic_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def on_fit_start(self):
        """Keep the non-registered frozen solver on Lightning's active device."""
        self.local_solver.device = self.device
        self.local_solver.model.to(self.device).eval()

    def training_step(self, batch: TensorDictBase, batch_idx: int):
        loss, mean_cost, best_cost = self._run_batch(batch, require_prob=True)
        batch_size = batch.batch_size[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/cost", mean_cost, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("train/best_cost", best_cost, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: TensorDictBase,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if dataloader_idx == 1:
            if (self.current_epoch + 1) % self.test_every_n_epochs != 0:
                return
            _, mean_cost, best_cost = self._run_batch(batch, require_prob=False)
            batch_size = batch.batch_size[0]
            self.log(
                "test/uniform_mean_cost",
                mean_cost,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self.log(
                "test/uniform_mean_best_cost",
                best_cost,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            return

        _, mean_cost, best_cost = self._run_batch(batch, require_prob=False)
        batch_size = batch.batch_size[0]
        self.log("val/cost", mean_cost, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("val/best_cost", best_cost, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def test_step(self, batch: TensorDictBase, batch_idx: int):
        _, mean_cost, best_cost = self._run_batch(batch, require_prob=False)
        batch_size = batch.batch_size[0]
        self.log("test/uniform_mean_cost", mean_cost, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log("test/uniform_mean_best_cost", best_cost, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def _run_batch(self, batch: TensorDictBase, require_prob: bool):
        batch = batch.to(self.device)
        losses = []
        mean_costs = []
        best_costs = []

        for instance_idx in range(batch.batch_size[0]):
            graph = self._build_graph(batch, instance_idx)
            heuristic = self.heuristic_net(graph)
            aco = DecompositionACO(
                heuristic=heuristic,
                n_ants=self.n_ants,
                max_subproblem_size=self.max_subproblem_size,
                decay=self.decay,
                alpha=self.alpha,
                beta=self.beta,
                close_weight=self.close_weight,
                device=self.device,
            )

            instance_costs = []
            for _ in range(self.aco_iterations):
                if require_prob:
                    decompositions, log_probs = aco.sample(require_prob=True)
                else:
                    decompositions = aco.sample()

                costs = self._evaluate_candidates(batch, instance_idx, decompositions)
                instance_costs.append(costs)
                aco.update_pheromone(decompositions, costs)

                if require_prob:
                    losses.append(self._reinforce_loss(log_probs, costs))

            all_costs = torch.cat(instance_costs)
            mean_costs.append(all_costs.mean())
            best_costs.append(all_costs.min())

        loss = (
            torch.stack(losses).mean()
            if losses
            else torch.zeros((), device=self.device)
        )
        return loss, torch.stack(mean_costs).mean(), torch.stack(best_costs).mean()

    @staticmethod
    def _reinforce_loss(log_probs: torch.Tensor, costs: torch.Tensor):
        """Leave-one-out baseline over ants of one large instance."""
        detached_costs = costs.detach()
        baseline = (detached_costs.sum() - detached_costs) / (len(costs) - 1)
        advantage = detached_costs - baseline
        return (advantage * log_probs).mean()

    def _build_graph(self, batch: TensorDictBase, instance_idx: int):
        locs = batch["locs"][instance_idx]
        n_nodes = locs.size(0)
        nodes = torch.arange(n_nodes, device=locs.device)
        source = nodes.repeat_interleave(n_nodes)
        target = nodes.repeat(n_nodes)
        keep = source != target
        edge_index = torch.stack((source[keep], target[keep]))
        distances = torch.cdist(locs, locs)
        edge_attr = distances[edge_index[0], edge_index[1]].unsqueeze(-1)
        return Data(x=locs, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def _evaluate_candidates(
        self,
        batch: TensorDictBase,
        instance_idx: int,
        decompositions: list[list[list[int]]],
    ):
        """Solve and merge all ants for one large instance."""
        groups_by_size: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
        merged_routes: list[list[list[int]]] = [[] for _ in decompositions]

        for ant_idx, decomposition in enumerate(decompositions):
            for group in decomposition:
                groups_by_size[len(group)].append((ant_idx, group))

        for _, entries in groups_by_size.items():
            instance_indices = torch.full(
                (len(entries),),
                instance_idx,
                dtype=torch.long,
                device=batch.device,
            )
            group_indices = torch.tensor(
                [group for _, group in entries],
                dtype=torch.long,
                device=batch.device,
            )

            # Each same-size group becomes one local problem. Grouping by size
            # lets the black-box model solve variable-size ACO groups in batch.
            source_td = batch[instance_indices]
            local_td = build_local_tensordict(
                source_td,
                group_indices.unsqueeze(1),
            )
            output = self.local_solver.solve(local_td, phase="test")
            local_actions = output["actions"]

            if local_actions.ndim != 2 or local_actions.size(0) != len(entries):
                raise RuntimeError(
                    "The local solver must return one action sequence per local group. "
                    "Use solver_num_starts=1 for decomposition training."
                )

            # NeuralLocalSolver moves local_td to its own device. Keep the
            # index mapping with its returned actions before gathering.
            group_indices = group_indices.to(local_actions.device)
            global_actions = local_actions_to_global(
                local_actions,
                group_indices.unsqueeze(1),
            )
            routes_per_group = actions_to_routes(
                global_actions,
                num_instances=len(entries),
                num_groups=1,
            )

            for (ant_idx, _), routes in zip(entries, routes_per_group):
                merged_routes[ant_idx].extend(routes)

        costs = []
        locs = batch["locs"][instance_idx]
        demand = batch[self.demand_key][instance_idx]
        capacity = batch["vehicle_capacity"][instance_idx]
        expected_customers = locs.size(0) - 1

        for routes in merged_routes:
            if self.global_improver is not None:
                routes = self.global_improver(routes, batch[instance_idx])
            metrics = evaluate_solution(
                routes=routes,
                locs=locs,
                demand=demand,
                vehicle_capacity=capacity,
                expected_customers=expected_customers,
            )
            cost = metrics.cost
            if not metrics.feasible:
                cost += self.infeasible_penalty
            costs.append(cost)

        return torch.tensor(costs, dtype=locs.dtype, device=self.device)
