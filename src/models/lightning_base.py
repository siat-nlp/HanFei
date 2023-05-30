"""
@Desc:
@Reference:
- from_argparse_args:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.utilities.argparse.html#p
ytorch_lightning.utilities.argparse.from_argparse_args
- ModelCheckpoint
https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/
pytorch_lightning.callbacks.ModelCheckpoint.html?highlight=ModelCheckpoint
-  Trainer.from_argparse_args(args)
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
- Optimizers: AdamW and AdaFactor
https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
Adaptive Learning Rates with Sublinear Memory Cost https://arxiv.org/abs/1804.04235
- optimizer_grouped_parameters
https://huggingface.co/transformers/v3.3.1/training.html
The optimizer allows us to apply different hyperpameters for specific parameter groups.
- rank_zero_only
http://www.liuxiao.org/2020/07/pytorch-lighting-%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%95%B4%E7%90%86/
使用 @rank_zero_only 修饰多线程中只在 RANK=0 调用的函数
- save vocab.json
https://huggingface.co/transformers/v1.0.0/model_doc/overview.html
@Notes:
- huggingface from_pretrained default store path
windows: ~/.cache/huggingface
- architecture
model.prepare_data()
initialize_distributed()
model.setup(stage)
model.train_dataloader()
model.val_dataloader()
model.test_dataloader()
"""

import argparse
import logging
import torch
import pytorch_lightning as pl

from pathlib import Path
from typing import Any, Dict, Optional
from pytorch_lightning.utilities import rank_zero_info

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from transformers.optimization import (
    Adafactor,
)

from src.configuration.constants import MODEL_CLASSES, GET_SCHEDULER_FUNCS
from src.utils.string_utils import are_same_strings

logger = logging.getLogger(__name__)


class BaseTransformer(pl.LightningModule):
    loss_names = {"loss", }
    metric_names = {"loss", }

    def __init__(
            self,
            hparams: argparse.Namespace,
            model_type="base",
            **kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.save_hyperparameters(hparams)  # save hparams to self.hparams
        self.output_dir = Path(self.hparams.output_dir)
        self.experiment_name = self.hparams.experiment_name
        self.model_name = self.hparams.model_name
        self.experiment_output_dir = self.output_dir.joinpath(self.hparams.experiment_name)
        self.pretrained_save_path = Path(self.experiment_output_dir).joinpath("best_tfmr")
        self.model_type = model_type
        self.batch_size = None  # for auto_scale_batch_size
        # record api
        self.config = None
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def _set_up(self,
                config: PretrainedConfig = None,
                tokenizer: PreTrainedTokenizer = None,
                model=None, **config_kwargs):
        # load pretrained settings
        # config
        self.config: PretrainedConfig = config if config is not None else \
            AutoConfig.from_pretrained(self.hparams.model_name_or_path,
                                       **config_kwargs)
        self._check_config(self.config)
        # tokenizer
        self.tokenizer: PreTrainedTokenizer = tokenizer if tokenizer is not None else \
            AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        # model
        self.model_class = MODEL_CLASSES[self.model_type]
        self.model = model if model is not None \
            else self._load_model(self.hparams.model_name_or_path, self.model_class, config)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def _check_config(self, config: PretrainedConfig):
        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(config, p), f"model config doesn't have a `{p}` attribute"
                setattr(config, p, getattr(self.hparams, p))

    def _load_model(self, model_name_or_path: str, model_class, config: PretrainedConfig = None, cache_dir=None):
        if config is None:
            return model_class.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        else:
            return model_class.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                config=config,
                trust_remote_code=True
            )

    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer, frequency=1):
        scheduler = None
        if self.hparams.lr_scheduler in GET_SCHEDULER_FUNCS:
            get_schedule_func = GET_SCHEDULER_FUNCS[self.hparams.lr_scheduler]
            scheduler = get_schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                          num_training_steps=self.total_steps())
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": frequency}
        elif are_same_strings(self.hparams.lr_scheduler, "ReduceLROnPlateau"):
            scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                         "interval": "epoch", "frequency": frequency, "monitor": "val_loss"}
        else:
            raise NotImplementedError()
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if are_same_strings(self.hparams.optimizer_class, "Adafactor"):
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        elif are_same_strings(self.hparams.optimizer_class, "AdamW"):
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        else:
            raise NotImplementedError(f"{self.hparams.optimizer_class} not available. Only Adafactor|Adafactor")

        self.optimizer = optimizer

        frequency = 1
        self.scheduler = self.get_lr_scheduler(self.optimizer, frequency=frequency)

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler
        }

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        if isinstance(self.hparams.accumulate_grad_batches, dict):
            accumulate_grad_batches = list(self.hparams.accumulate_grad_batches.values())[-1]
        else:
            accumulate_grad_batches = self.hparams.accumulate_grad_batches
        effective_batch_size = self.hparams.train_batch_size * accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, stage: Optional[str] = None):
        # stage: train, test, eval
        self.dataset_size = len(self.train_dataloader().dataset)

    def get_dataloader(self, data_type: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size, shuffle=False)

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.pretrained_save_path
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"hugging face format checkpoint save at {save_path}")

    def update_loss_names(self, loss_result: Dict, update_flag=True):
        if not update_flag:
            return
        for loss_name_ in loss_result.keys():
            if loss_name_ not in self.loss_names:
                self.loss_names.add(loss_name_)

    def update_metric_names(self, metrics: Dict, update_flag=True):
        if not update_flag:
            return
        for metric_name_ in metrics.keys():
            if metric_name_ not in self.metric_names:
                self.metric_names.add(metric_name_)
