import os
import random
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate

@dataclass
class Main:
    proj_name: str
    seed: int
    debug: bool
    exp: object
    mlflow: object

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):
    # Init RNG_level s
    random.seed(cfg.seed)
    cfg = instantiate(cfg)

    # Run experiment
    if cfg.debug > 0:
        os.environ["HYDRA_FULL_ERROR"] = "1" # In debug mode
        cfg.exp.run(cfg)
    else:
        cfg.exp.main(cfg)

if __name__ == "__main__":
    main()
