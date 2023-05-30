"""
@Desc:
@Reference:
@Notes:
"""

import sys
import glob
import os
from pathlib import Path

import pytorch_lightning as pl

from src.configuration.task.config_args import parse_args_for_config
from src.utils.wrapper import print_done
from src.utils.string_utils import are_same_strings
from src.models.basic_trainer import BasicTrainer
from src.modules.pl_callbacks import Seq2SeqLoggingCallback, Seq2SeqCheckpointCallback


class BasicPLTrainer(BasicTrainer):
    def __init__(self, args, trainer_name="basic-pl-trainer"):
        # parameters
        super().__init__(args, trainer_name=trainer_name)

        # customized variables
        self.pl_trainer = None
        self.device = self.model.device if hasattr(self.model, "device") else self.device

    @print_done(desc="Creating directories and fix random seeds")
    def _init_args(self, args):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        pl.seed_everything(args.seed, workers=True)  # reproducibility

    @print_done(desc="initialize model")
    def _init_model(self, args):
        # automatically download from huggingface
        print(f"model_path: {args.model_name_or_path}")
        raise NotImplementedError(f"args.model_name: {args.model_name}")

    @print_done("set up pytorch lightning trainer")
    def _init_pl_trainer(self, args, model, logger):
        extra_callbacks = []
        self.checkpoint_callback = Seq2SeqCheckpointCallback(
            output_dir=self.save_dir,
            experiment_name=self.experiment_name,
            monitor="val_loss",
            save_top_k=args.save_top_k,
            every_n_train_steps=args.every_n_train_steps,
            verbose=args.ckpt_verbose,
        )

        # initialize pl_trainer
        if args.gpus is not None and args.gpus > 1:
            self.train_params["distributed_backend"] = "ddp"

        self.pl_trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args,
            enable_model_summary=False,
            callbacks=[self.checkpoint_callback, Seq2SeqLoggingCallback(), pl.callbacks.ModelSummary(max_depth=1)]
                      + extra_callbacks,
            logger=logger,
            **self.train_params,
        )

    @property
    def checkpoints(self):
        return list(sorted(glob.glob(os.path.join(self.save_dir, "*.ckpt"), recursive=True)))

    def auto_find_lr_rate(self):
        if self.pl_trainer.auto_lr_find:
            self.pl_trainer.tune(self.model)
            print(f"after tuning: {self.model.learning_rate}")

            # 开始寻找合适的学习率
            lr_finder = self.model.tuner.lr_find(self.model)
            # 展示loss和学习率的曲线
            print(f"auto find the best learning rate of model {self.model.model_name}:\n{lr_finder.results}")
            # 设置为推荐的学习率
            suggested_lr = lr_finder.suggestion()
            print(f"the suggested lr: {suggested_lr}")
            self.model.hyparams.learning_rate = suggested_lr

    def auto_find_batch_size(self):
        if self.pl_trainer.auto_scale_batch_size == "binsearch":
            self.pl_trainer.tune(self.model)
            print(f"auto find the best of batch size of {self.model.model_name}:\n{self.pl_trainer.batch_size}")
            self.model.hyparams.train_batch_size = self.model.batch_size

    def train(self):
        self.auto_find_lr_rate()
        self.auto_find_batch_size()

        self.pl_trainer.logger.log_hyperparams(self.args)

        if self.checkpoints:
            # training
            best_ckpt = self.checkpoints[-1]
            self.pl_trainer.fit(self.model, ckpt_path=best_ckpt)
        else:
            # training
            if hasattr(self.model, "init_for_vanilla_weights"):
                self.model.init_for_vanilla_weights()
            self.pl_trainer.fit(self.model)
