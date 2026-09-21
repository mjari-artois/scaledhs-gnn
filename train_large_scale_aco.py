from __future__ import annotations

import math

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from ai4co_gnn.large_scale.aco.trainer import (
    ACODecompositionTrainer,
    DecompositionDataModule,
)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, strict: bool):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)

    for prefix in ("", "model.", "module."):
        candidate = (
            state_dict
            if not prefix
            else {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        )
        if not candidate:
            continue
        try:
            model.load_state_dict(candidate, strict=strict)
            return
        except RuntimeError:
            pass

    raise RuntimeError(f"Could not load solver checkpoint: {checkpoint_path}")


def configure_hardware(cfg: DictConfig) -> int:
    """Scale the campaign from the CUDA devices visible to this process."""
    if not cfg.hardware.auto_configure:
        return int(cfg.trainer.devices) if isinstance(cfg.trainer.devices, int) else 1

    cuda_devices = torch.cuda.device_count()
    if cuda_devices:
        accelerator = "gpu"
        devices = cuda_devices
        strategy = "ddp" if devices > 1 else "auto"
        precision = "bf16-mixed"
        torch.set_float32_matmul_precision(cfg.hardware.matmul_precision)
    elif torch.backends.mps.is_available():
        accelerator, devices, strategy, precision = "mps", 1, "auto", "32"
    else:
        accelerator, devices, strategy, precision = "cpu", 1, "auto", "32"

    per_device_batch = cfg.hardware.per_device_batch_size
    effective_batch = per_device_batch * devices
    accumulate_grad_batches = max(
        1,
        math.ceil(cfg.hardware.target_effective_batch_size / effective_batch),
    )
    n_ants = min(
        cfg.hardware.max_ants,
        max(cfg.hardware.min_ants, cfg.hardware.ants_per_device * devices),
    )

    with open_dict(cfg):
        cfg.trainer.accelerator = accelerator
        cfg.trainer.devices = devices
        cfg.trainer.strategy = strategy
        cfg.trainer.precision = precision
        cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
        cfg.data.batch_size = per_device_batch
        cfg.data.num_workers = cfg.hardware.num_workers_per_device
        cfg.data.train_data_size = cfg.hardware.train_instances_per_device * devices
        cfg.aco.n_ants = n_ants

    return devices


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="large_scale/aco_train.yaml",
)
def main(cfg: DictConfig):
    devices = configure_hardware(cfg)
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    if devices > 1:
        print(
            f"Using {devices} GPUs with DDP: per-device batch={cfg.data.batch_size}, "
            f"gradient accumulation={cfg.trainer.accumulate_grad_batches}, "
            f"ants={cfg.aco.n_ants}"
        )

    env = instantiate(cfg.env)
    local_solver_model = instantiate(cfg.model, env=env)
    load_checkpoint(
        local_solver_model,
        to_absolute_path(cfg.solver.checkpoint_path),
        strict=cfg.solver.checkpoint_strict,
    )

    module = ACODecompositionTrainer(
        heuristic_net=instantiate(cfg.heuristic_net),
        local_solver_model=local_solver_model,
        env=env,
        **OmegaConf.to_container(cfg.aco, resolve=True),
    )
    data_module = DecompositionDataModule(
        train_source=cfg.data.train_source,
        validation_source=cfg.data.validation_source,
        test_source=cfg.data.test_source,
        train_path=(
            to_absolute_path(cfg.data.train_path)
            if cfg.data.train_path is not None
            else None
        ),
        validation_path=(
            to_absolute_path(cfg.data.validation_path)
            if cfg.data.validation_path is not None
            else None
        ),
        test_path=(
            to_absolute_path(cfg.data.test_path)
            if cfg.data.test_path is not None
            else None
        ),
        train_data_size=cfg.data.train_data_size,
        validation_data_size=cfg.data.validation_data_size,
        test_data_size=cfg.data.test_data_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        dataset_format=cfg.data.dataset_format,
        env=env,
    )

    if cfg.smoke_test:
        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            precision=cfg.trainer.precision,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )
    else:
        if cfg.wandb.enabled:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                group=cfg.wandb.group,
                tags=list(cfg.wandb.tags),
                save_dir=to_absolute_path(cfg.wandb.save_dir),
                offline=cfg.wandb.offline,
                log_model=False,
            )
            # Lightning logs hyperparameters only on rank zero under DDP.
            logger.log_hyperparams(
                OmegaConf.to_container(cfg, resolve=True),
            )
        else:
            logger = False
        trainer = instantiate(cfg.trainer, logger=logger)

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
