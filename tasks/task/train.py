"""
@Desc:
@Reference:
- logger and WandLogger
Weights and Biases is a third-party logger
https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html
@Notes:

"""
import sys
import os
import pytorch_lightning as pl

from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.task.config_args import parse_args_for_config
from src.models.task import gpt_code, llama_finetune
from src.utils.wrapper import print_done
from src.utils.string_utils import are_same_strings
from src.models.basic_pl_trainer import BasicPLTrainer
from src.modules.pl_callbacks import Seq2SeqLoggingCallback, Seq2SeqCheckpointCallback, SaveCheckpointEveryEpoch

print("starting", flush=True)


class EventTriggerTrainer(BasicPLTrainer):
    def __init__(self, args, trainer_name="event-trigger-trainer"):
        # parameters
        super().__init__(args, trainer_name=trainer_name)

        self._init_model(self.args)
        self._init_logger(self.args, self.model)
        self._init_pl_trainer(self.args, self.model, self.logger)

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
        # automatically download from huggingface project
        print(f"model_path: {args.model_name_or_path}")
        # ============= gpt ===============
        if are_same_strings(args.model_name, "gpt"):
            self.model: gpt_code = gpt_code(args)
        # ============= llama ===============
        elif are_same_strings(args.model_name, "llama"):
            self.model: llama_finetune = llama_finetune(args)
        else:
            raise NotImplementedError(f"args.model_name: {args.model_name}")

    @print_done("set up pytorch lightning trainer")
    def _init_pl_trainer(self, args, model, logger):
        extra_callbacks = []
        # self.checkpoint_callback = Seq2SeqCheckpointCallback(
        #     output_dir=self.save_dir,
        #     experiment_name=self.experiment_name,
        #     monitor="val_loss",
        #     save_top_k=args.save_top_k,
        #     every_n_train_steps=args.every_n_train_steps,
        #     verbose=args.ckpt_verbose,
        # )

        # initialize pl_trainer
        # if args.gpus is not None and args.gpus > 1:
        #     self.train_params["distributed_backend"] = "ddp"

        self.checkpoint_callback = SaveCheckpointEveryEpoch(
            output_dir=self.save_dir,
            experiment_name=self.experiment_name,
            last_k=args.save_top_k,
            save_weights_only=False,
            every_n_epochs=args.save_every_n_epochs,
            every_n_train_steps=args.save_every_n_steps
        )

        self.pl_trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            args,
            enable_model_summary=False,
            callbacks=[self.checkpoint_callback, Seq2SeqLoggingCallback(), pl.callbacks.ModelSummary(max_depth=1)]
                      + extra_callbacks,
            logger=logger,
            **self.train_params,
        )

    def train(self):
        self.auto_find_lr_rate()
        self.auto_find_batch_size()

        self.pl_trainer.logger.log_hyperparams(self.args)

        if self.checkpoints:
            # training
            best_ckpt = self.checkpoints[-1]
            self.pl_trainer.fit(self.model, ckpt_path=best_ckpt)
        else:
            self.pl_trainer.fit(self.model)


if __name__ == '__main__':
    hparams = parse_args_for_config()
    trainer = EventTriggerTrainer(hparams)
    trainer.train()
