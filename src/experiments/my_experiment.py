import hydra
import dagshub
import mlflow
import logging


class MyExp:
    def __init__(self):
        self.mlflow_id = 0

    def setup(self, proj_name, debug:bool):
        # MLFlow setup
        mlflow.start_run(experiment_id=self.mlflow_id, 
                         run_name='debug' if not debug else None)
        mlflow_username = hydra.core.hydra_config.HydraConfig.get().job.env_set.MLFLOW_TRACKING_USERNAME
        dagshub.init(proj_name, mlflow_username, mlflow=True)

    def end_run(self, seed:int):
        mlflow.log_param('seed', seed)
        mlflow.end_run()

    def run_experiment(self, cfg):
        logging.info(f"Running with seed: {cfg.seed}")
        self.setup(cfg.proj_name, cfg.debug)
        self.end_run(cfg.seed)
