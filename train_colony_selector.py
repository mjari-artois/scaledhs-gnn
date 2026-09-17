"""Hydra entry point for training the V3 colony neighborhood selector."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

from ai4co_gnn.large_scale.colony_data import (
    episodes_from_tensordict,
    make_synthetic_episodes,
    sequential_repair,
    TensorDictEpisodeDataset,
)
from ai4co_gnn.large_scale.data import load_uniform_npz
from ai4co_gnn.large_scale.local_solver import NeuralLocalSolver
from solve_large_cvrp import _load_checkpoint
from ai4co_gnn.large_scale.selector import GNNColonySelector
from ai4co_gnn.large_scale.train_colony import (
    ColonySelectorLightning,
    build_colony_dataloader,
)


@hydra.main(version_base="1.3", config_path="configs", config_name="colony/train.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        pl.seed_everything(int(cfg.seed), workers=True)

    routing_env = instantiate(cfg.env)
    if bool(cfg.routing.use_sequential_fallback):
        repair_fn = sequential_repair
    else:
        frozen_model = instantiate(cfg.model, env=routing_env)
        checkpoint_path = to_absolute_path(str(cfg.routing.checkpoint_path))
        _load_checkpoint(
            frozen_model,
            checkpoint_path,
            strict=bool(cfg.routing.checkpoint_strict),
        )
        frozen_solver = NeuralLocalSolver(
            model=frozen_model,
            env=routing_env,
            device=None if str(cfg.routing.device) == "auto" else str(cfg.routing.device),
            num_starts=int(cfg.routing.num_starts),
        )

        def repair_fn(local_td):
            return frozen_solver.solve(
                local_td,
                phase=str(cfg.routing.phase),
            )

    if cfg.data.train_path:
        if not cfg.data.validation_path:
            raise ValueError("data.validation_path is required when train_path is set")
        train_episodes = TensorDictEpisodeDataset(
            load_uniform_npz(to_absolute_path(str(cfg.data.train_path))),
            max_instances=int(cfg.data.train_size), seed=int(cfg.seed or 0),
        )
        validation_episodes = TensorDictEpisodeDataset(
            load_uniform_npz(to_absolute_path(str(cfg.data.validation_path))),
            max_instances=int(cfg.data.validation_size), seed=int(cfg.seed or 0) + 100_000,
        )
    elif bool(cfg.data.generate):
        routing_env.generator.variant_preset = str(cfg.data.variant)
        train_data = routing_env.generator(int(cfg.data.train_size))
        validation_data = routing_env.generator(int(cfg.data.validation_size))
        train_episodes = TensorDictEpisodeDataset(
            train_data, seed=int(cfg.seed or 0)
        )
        validation_episodes = TensorDictEpisodeDataset(
            validation_data, seed=int(cfg.seed or 0) + 100_000
        )
    else:
        train_episodes = make_synthetic_episodes(
            int(cfg.data.train_size),
            int(cfg.data.num_customers),
            capacity=float(cfg.data.capacity),
            seed=int(cfg.seed or 0),
        )
        validation_episodes = make_synthetic_episodes(
            int(cfg.data.validation_size),
            int(cfg.data.num_customers),
            capacity=float(cfg.data.capacity),
            seed=int(cfg.seed or 0) + 100_000,
        )

    selector = GNNColonySelector(**OmegaConf.to_container(cfg.selector, resolve=True))
    optimization = OmegaConf.to_container(cfg.optimization, resolve=True)
    batch_size = int(optimization.pop("batch_size"))
    module = ColonySelectorLightning(
        selector=selector,
        repair_fn=repair_fn,
        **OmegaConf.to_container(cfg.search, resolve=True),
        **optimization,
    )

    logger = False
    if bool(cfg.wandb.enabled):
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=str(cfg.wandb.project),
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            offline=bool(cfg.wandb.offline),
            save_dir=to_absolute_path(str(cfg.paths.log_dir)),
            log_model=False,
        )
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True),
            allow_val_change=True,
        )

    checkpoint_dir = Path(to_absolute_path(str(cfg.paths.checkpoint_dir)))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="selector-{epoch:03d}-{val_gap:.5f}",
        monitor="val/gap",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        deterministic=bool(cfg.trainer.deterministic),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        gradient_clip_algorithm=str(cfg.trainer.gradient_clip_algorithm),
        check_val_every_n_epoch=int(cfg.trainer.check_val_every_n_epoch),
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
        enable_model_summary=bool(cfg.trainer.enable_model_summary),
        enable_checkpointing=bool(cfg.trainer.enable_checkpointing),
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=int(cfg.trainer.num_sanity_val_steps),
        logger=logger,
        callbacks=[checkpoint],
    )
    trainer.fit(
        module,
        train_dataloaders=build_colony_dataloader(
            train_episodes,
            batch_size=batch_size,
            num_workers=int(cfg.data.num_workers),
        ),
        val_dataloaders=build_colony_dataloader(
            validation_episodes,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
        ),
        ckpt_path=(
            to_absolute_path(str(cfg.resume_checkpoint))
            if cfg.resume_checkpoint else None
        ),
    )

    # Final held-out inference on the supplied uniform-500 test set.  This is
    # deliberately outside the optimization loop: test instances never affect
    # selector gradients or the training baseline.
    if cfg.data.test_path:
        test_data = load_uniform_npz(to_absolute_path(str(cfg.data.test_path)))
        test_episodes = episodes_from_tensordict(
            test_data,
            max_instances=int(cfg.data.test_size),
            seed=int(cfg.seed or 0) + 200_000,
        )
        module.selector.eval()
        costs: list[float] = []
        gaps: list[float] = []
        with torch.no_grad():
            for episode in test_episodes:
                search = module._new_search(episode)
                result = search.run()
                costs.append(float(result["cost"]))
                if episode.reference_cost is not None:
                    gaps.append(
                        (float(result["cost"]) - episode.reference_cost)
                        / max(episode.reference_cost, 1e-8)
                    )
        test_cost = sum(costs) / max(len(costs), 1)
        test_gap = sum(gaps) / max(len(gaps), 1)
        print(
            f"test_instances={len(costs)} "
            f"test_mean_cost={test_cost:.6f} "
            f"test_mean_gap={test_gap:.6f}"
        )
        if logger is not False:
            logger.experiment.log(
                {
                    "test/cost": test_cost,
                    "test/gap": test_gap,
                    "test/instances": len(costs),
                }
            )
    print(f"best_checkpoint={checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
