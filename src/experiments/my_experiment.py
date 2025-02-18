import hydra
import dagshub
import mlflow
import logging
from pathlib import Path


class MyExp:
    def __init__(self):
        self.mlflow_id = 0
        self.exp_name = "MyExp"

    def get_config(self):
        return {
                'exp_name': self.exp_name
                }

    def setup(self, proj_name, username, mlflow_pass, debug:bool):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'

        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        dagshub.init(proj_name, username, mlflow=not debug)
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass
        mlflow.start_run(experiment_id=self.mlflow_id)

    def end_run(self, seed:int):
        mlflow.log_param('success', True)
        mlflow.log_param('seed', seed)
        mlflow.log_params(self.get_config)
        mlflow.log_artifact(self.out_dir/self.log_filename)

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def end_failed_run(self, error, seed:int):
            logging.info("Failed!")
            logging.info(error)
            mlflow.log_param('success', False)

            mlflow.log_param('seed', seed)
            mlflow.log_artifact(self.out_dir/self.log_filename)
            mlflow.log_artifact(self.overrides_config)

            mlflow.end_run()
            finished_run = mlflow.get_run(self.run.info.run_id)
            logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.proj_name, cfg.username, cfg.mlflow_pass, cfg.debug)
        self.end_run(cfg.seed)

    def main(self, cfg):
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failed_run(e, cfg.seed)
