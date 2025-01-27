import os
import torch
import hydra
import dagshub
import mlflow
import logging
from pathlib import Path

from utils.nn import train_epoch, eval_model


class Training:
    def __init__(self):
        self.mlflow_id = 1 # 1 - training
        self.exp_name = "Training"

    def setup(self, partial_model, optim, loader, device):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                           out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())

    def start_run(self, proj_name, username, mlflow_pass, run_name, debug_level:int):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{self.out_dir}/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        dagshub.init(proj_name, username, mlflow=(debug_level != 3))
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass

        (mlflow_id, run_name) = (self.mlflow_id, run_name) if debug_level == 0 else (0, 'debug')
        mlflow.start_run(experiment_id=(self.mlflow_id), run_name=run_name)

        self.run = mlflow.active_run()
        logging.info(f"MLFow run ID: {self.run.info.run_id}, status: {self.run.info.status}")

    def end_run(self, metrics, seed:int):
        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv')

        # Log model
        self.model.to('cpu')
        torch.save(self.model, self.out_dir/'model.pt') # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints
        mlflow.log_artifact(self.out_dir/'model.pt')
        mlflow.log_artifact(self.out_dir/'state_dict.pt')

        # Log base
        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def run_experiment(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.optim, cfg.loader,
                   cfg.device)
        self.start_run(cfg.proj_name, cfg.username, cfg.mlflow_pass, 
                       cfg.run_name, cfg.debug_level)

        # Param. logging
        mlflow.log_params({
            'depth': self.model.depth.item(),
            'leaf_width': self.model.leaf_width,
            'task': cfg.loader.name,
            'epochs': cfg.epochs,
            })

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(self.model, self.optim, cfg.loader.train, self.criterion, epoch, cfg.device)
            val_loss, val_acc = eval_model(self.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
            test_loss, test_acc = eval_model(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            self.model.to(cfg.device)

        # # Testing
        test_loss, test_acc= eval_model(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            })

        self.end_run(metrics, cfg.seed)
