import hydra
import dagshub
import mlflow
import logging
from pathlib import Path


class MyExp:
    def __init__(self):
        self.mlflow_id = 0
        self.exp_name = "MyExp"

    def setup(self, proj_name, username, mlflow_pass, debug:bool):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        dagshub.init(proj_name, username, mlflow=not debug)
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass
        mlflow.start_run(experiment_id=self.mlflow_id)

    def end_run(self, seed:int):
        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.end_run()

    def run_experiment(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.proj_name, cfg.username, cfg.mlflow_pass, cfg.debug)
        self.end_run(cfg.seed)
