import os

from typing import Optional

import hydra
import lightning as L
import pyrootutils

from omegaconf import DictConfig
from rl4co import utils
from rl4co.data.utils import save_tensordict_to_npz

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def generate(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)
    for type in ["val", "test"]:
        print(f"{type}_file : [", end="")
        for variant_preset in env.generator.available_variants():
            num_loc = env.generator.num_loc
            path = f"data/{variant_preset}/{type}_{num_loc}.npz"
            if os.path.exists(path):
                print(
                    f"'{variant_preset}': '{variant_preset}/{type}_{num_loc}.npz'",
                    end=" ",
                )
                continue
            env.generator.variant_preset = variant_preset
            td = env.generator(getattr(cfg.model, f"{type}_data_size"))
            os.makedirs(f"data/{variant_preset}", exist_ok=True)
            save_tensordict_to_npz(
                td, f"data/{variant_preset}/{type}_{num_loc}.npz", compress=True
            )
            print(
                f"'{variant_preset}': '{variant_preset}/{type}_{num_loc}.npz',", end=" "
            )
        print("]")


if __name__ == "__main__":
    generate()
