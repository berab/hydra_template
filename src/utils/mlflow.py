import os
import logging
from pathlib import Path
import mlflow
import dagshub

class MLFlow:
    def __init__(self, username: str, token:str, mlflow_id: int, run_name: str, 
                 active: bool):
        self.token = token
        self.mlflow_id = mlflow_id 
        self.run_name = run_name
        self.active = active
        self.username = username

    def start(self):
        dagshub.init(proj_name, self.username, mlflow=self.active)
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = self.token
        mlflow.start_run(experiment_id=self.mlflow_id, run_name=self.run_name)

        run = mlflow.active_run()
        logging.info(f"MLFlow run ID: {run.info.run_id}, status: {run.info.status}")
