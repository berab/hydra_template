import os
import random
import hydra
import mlflow
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
from hydra.experimental.callback import Callback
from typing import Any

class MLFlowToCSV(Callback):
    def __init__(self, exp):
        self.exp = exp

    def on_multirun_end(self, config: DictConfig, **kwargs: Any):
        mlflow.set_tracking_uri("file:data/mlruns")
        experiment = mlflow.get_experiment_by_name(self.exp)

        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        os.makedirs("data/csv_files", exist_ok=True)
        runs_df.to_csv(f"data/csv_files/{self.exp}.csv", index=False)

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
