"""
@Desc:
@Reference:
@Notes:
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # 在tasks文件夹中可以直接运行程序

from pytorch_lightning.loggers import CSVLogger
from src.utils.wrapper import print_done


class BasicTrainer(object):
    def __init__(self, args, trainer_name: str = "basic_trainer"):
        # parameters
        self.trainer_name = trainer_name
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.experiment_name = args.experiment_name
        self.experiment_output_dir = self.output_dir.joinpath(self.experiment_name)
        self.save_dir = self.experiment_output_dir.joinpath("checkpoints")
        self.data_dir = args.data_dir
        self.log_dir = self.experiment_output_dir.joinpath("logs")
        self.cache_dir = self.experiment_output_dir.joinpath("cache")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.logger = None
        self.tokenizer = None
        self.vocab = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.criterion = None
        self.model = None
        self.train_params = {}

        self._init_args(self.args)

    def _init_logger(self, args, model):
        """
        - WandbLogger
        name: Display name for the run.
        project: The name of the project to which this run will belong.
        """
        if args.logger_name == "CSVLogger":
            self.logger = CSVLogger(save_dir=self.log_dir, name=f'{self.experiment_name}_CSVLogger')
        elif args.logger_name == "WandbLogger":
            from pytorch_lightning.loggers import WandbLogger
            os.environ["WANDB_API_KEY"] = 'xxxxxxxxxxxxxx'  # TODO your api key

            self.logger = WandbLogger(name=f'{self.experiment_name}_WandLogger', project=self.experiment_name)
        else:
            raise NotImplementedError

    @print_done(desc="Creating directories and fix random seeds")
    def _init_args(self, args):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
