import torch

from ai4co_gnn.large_scale.colony_data import make_synthetic_episodes, sequential_repair
from ai4co_gnn.large_scale.selector import GNNColonySelector
from ai4co_gnn.large_scale.train_colony import train_colony_selector


def test_colony_selector_trains_for_ten_epochs():
    episodes = make_synthetic_episodes(
        count=2,
        num_customers=6,
        capacity=1.0,
        seed=123,
    )
    validation = make_synthetic_episodes(
        count=1,
        num_customers=6,
        capacity=1.0,
        seed=456,
    )
    selector = GNNColonySelector(units=16, depth=1, graph_k=3)

    module = train_colony_selector(
        selector=selector,
        repair_fn=sequential_repair,
        episodes=episodes,
        validation_episodes=validation,
        max_epochs=10,
        batch_size=1,
        rounds=1,
        n_ants=1,
        max_local_customers=50,
        candidate_pool_size=8,
        trainer_kwargs={
            "logger": False,
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "accelerator": "cpu",
            "devices": 1,
        },
    )

    assert module.trainer.current_epoch == 10
    assert torch.isfinite(module.running_baseline)
    assert module.baseline_initialized.item()

