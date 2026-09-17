from __future__ import annotations

import json
from pathlib import Path
import time

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from rl4co.utils import get_pylogger

from ai4co_gnn.large_scale.coordinator import AngularSubproblemCoordinator
from ai4co_gnn.large_scale.data import load_uniform_npz
from ai4co_gnn.large_scale.solution import actions_to_routes, evaluate_solution

log = get_pylogger(__name__)


def _load_checkpoint(model, checkpoint_path: str, strict: bool = True):
    """Load a Lightning or plain state-dict checkpoint into the model."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict")

    # Support checkpoints saved with common wrapper prefixes.
    candidates = [state_dict]
    for prefix in ("model.", "module."):
        candidates.append(
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        )

    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            model.load_state_dict(candidate, strict=strict)
            return model
        except RuntimeError as error:
            last_error = error

    raise RuntimeError(
        f"Could not load checkpoint '{checkpoint_path}'."
    ) from last_error


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="large_scale/cvrp.yaml",
)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = instantiate(cfg.env)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model, env=env)
    _load_checkpoint(
        model,
        to_absolute_path(cfg.checkpoint_path),
        strict=cfg.checkpoint_strict,
    )
    model = model.to(device).eval()

    data_path = to_absolute_path(cfg.data_path)
    log.info(f"Loading data from {data_path}")
    if cfg.dataset_format == "uniform_npz":
        global_td = load_uniform_npz(data_path)
    elif cfg.dataset_format == "rl4co_npz":
        global_td = env.load_data(data_path)
    else:
        raise ValueError(
            f"Unknown dataset_format={cfg.dataset_format}. "
            "Use 'uniform_npz' or 'rl4co_npz'."
        )
    global_td = global_td.to(device)

    coordinator = AngularSubproblemCoordinator(
        model=model,
        env=env,
        group_size=cfg.group_size,
        num_starts=cfg.num_starts,
        device=device,
        demand_key=cfg.demand_key,
        max_vehicles=cfg.max_vehicles,
        refinement_enabled=cfg.refinement.enabled,
        max_candidates=cfg.refinement.max_candidates,
    )

    wandb_logger = None
    if cfg.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags),
            save_dir=to_absolute_path(cfg.wandb.save_dir),
            offline=cfg.wandb.offline,
            log_model=False,
        )
        wandb_logger.experiment.config.update(
            {
                "dataset_path": data_path,
                "dataset_format": cfg.dataset_format,
                "num_customers": cfg.num_customers,
                "group_size": cfg.group_size,
                "num_starts": cfg.num_starts,
                "refinement_enabled": cfg.refinement.enabled,
                "refinement_max_candidates": cfg.refinement.max_candidates,
                "checkpoint_path": to_absolute_path(cfg.checkpoint_path),
                "device": str(device),
            },
            allow_val_change=True,
        )
        log.info(
            f"W&B run: project={cfg.wandb.project} name={cfg.wandb.name}"
        )

    # Measure the full test pass: batched model inference, route reconstruction,
    test_start_time = time.perf_counter()
    result = coordinator.solve(global_td, phase="test")
    actions = result.global_actions

    num_instances = global_td.batch_size[0]
    num_groups = result.local_indices.shape[1]
    solutions = actions_to_routes(
        global_actions=actions,
        num_instances=num_instances,
        num_groups=num_groups,
    )

    demand_key = cfg.demand_key
    if demand_key not in global_td.keys():
        raise KeyError(
            f"Demand field '{demand_key}' was not found. "
            f"Available fields: {list(global_td.keys())}"
        )

    saved_instances = []

    for instance_idx, routes in enumerate(solutions):
        metrics = evaluate_solution(
            routes=routes,
            locs=global_td["locs"][instance_idx],
            demand=global_td[demand_key][instance_idx],
            vehicle_capacity=global_td["vehicle_capacity"][instance_idx],
            expected_customers=cfg.num_customers,
            max_vehicles=cfg.max_vehicles,
        )

        log.info(
            f"instance={instance_idx} routes={len(routes)} "
            f"cost={metrics.cost:.4f} feasible={metrics.feasible} "
            f"served={metrics.served_customers}/{metrics.expected_customers}"
        )

        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "instance/cost": metrics.cost,
                    "instance/feasible": int(metrics.feasible),
                    "instance/route_count": len(routes),
                    "instance/served_customers": metrics.served_customers,
                },
                step=instance_idx,
            )

        saved_instances.append(
            {
                "routes": routes,
                "cost": metrics.cost,
                "feasible": metrics.feasible,
                "served_customers": metrics.served_customers,
                "expected_customers": metrics.expected_customers,
                "duplicate_customers": metrics.duplicate_customers,
                "missing_customers": metrics.missing_customers,
                "capacity_violations": metrics.capacity_violations,
            }
        )

        if not metrics.feasible:
            log.warning(
                f"instance={instance_idx} duplicates={metrics.duplicate_customers} "
                f"missing={metrics.missing_customers} "
                f"capacity_violations={metrics.capacity_violations}"
            )

    test_runtime_seconds = time.perf_counter() - test_start_time
    seconds_per_instance = test_runtime_seconds / len(saved_instances)

    solution_path = Path(to_absolute_path(cfg.solution_output_path))
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    with solution_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "data_path": data_path,
                "dataset_format": cfg.dataset_format,
                "group_size": cfg.group_size,
                "refinement_enabled": cfg.refinement.enabled,
                "refinement_max_candidates": cfg.refinement.max_candidates,
                "mean_cost": sum(item["cost"] for item in saved_instances)
                / len(saved_instances),
                "num_instances": len(saved_instances),
                "num_feasible": sum(
                    item["feasible"] for item in saved_instances
                ),
                "test_runtime_seconds": test_runtime_seconds,
                "seconds_per_instance": seconds_per_instance,
                "instances": saved_instances,
            },
            file,
            indent=2,
        )

    log.info(f"Saved routes and metrics to {solution_path}")
    mean_cost = sum(item["cost"] for item in saved_instances) / len(
        saved_instances
    )
    feasible_count = sum(item["feasible"] for item in saved_instances)
    log.info(
        f"SUMMARY instances={len(saved_instances)} "
        f"feasible={feasible_count}/{len(saved_instances)} "
        f"mean_cost={mean_cost:.4f} "
        f"test_runtime_seconds={test_runtime_seconds:.3f} "
        f"seconds_per_instance={seconds_per_instance:.3f}"
    )

    if wandb_logger is not None:
        wandb_logger.log_metrics(
            {
                "summary/mean_cost": mean_cost,
                "summary/feasible_instances": feasible_count,
                "summary/num_instances": len(saved_instances),
                "summary/feasibility_rate": feasible_count / len(saved_instances),
                "summary/test_runtime_seconds": test_runtime_seconds,
                "summary/seconds_per_instance": seconds_per_instance,
            },
            step=len(saved_instances),
        )
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
