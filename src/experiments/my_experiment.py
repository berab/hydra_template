import hydra
import dagshub
import mlflow
import logging
from pathlib import Path

class MyExp:
    def __init__(self):
        self.exp_name = "MyExp"
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

    def get_config(self):
        return {
                'exp_name': self.exp_name,
                }

    def setup(self, proj_name, my_mlflow):
        # MLFlow setup
        my_mlflow.start()

    def end_run(self, seed:int):
        mlflow.log_param('success', True)
        mlflow.log_param('seed', seed)
        mlflow.log_params(self.get_config())
        mlflow.log_artifact(self.out_dir/self.log_filename)

        run = mlflow.active_run()
        mlflow.end_run()
        finished_run = mlflow.get_run(run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def end_failed_run(self, error, seed:int):
        logging.info("Failed!")
        logging.info(error)
        mlflow.log_param('success', False)

        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        run = mlflow.active_run()
        mlflow.end_run()
        finished_run = mlflow.get_run(run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.proj_name, cfg.mlflow)
        self.end_run(cfg.seed)

    def main(self, cfg):
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failed_run(e, cfg.seed)
