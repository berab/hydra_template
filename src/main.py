import torch
import random
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate

@dataclass
class Main:
    proj_name: str
    username: str
    mlflow_pass: str
    seed: int
    debug_level: int
    exp: object
    run_name: str
    device: torch.device

    epochs: int
    model: object
    optim: object
    loader: object

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):
    # Init RNGs
    print(cfg.device)
    random.seed(cfg.seed)
    cfg = instantiate(cfg)
    cfg.exp.run_experiment(cfg)

if __name__ == "__main__":
    main()
