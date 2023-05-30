"""
@Desc:
@Reference:
@Notes:.
- pytorch_lightning
https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html
- ModelCheckpoint
If we want set ModelCheckpoint with save_top_k, we need set callback_metrics for trainer.
- rank_zero_only
Whether the value will be logged only on rank 0. This will prevent synchronization which
would produce a deadlock as not all processes would perform this log call.
"""
import os
import pytorch_lightning as pl
import torch

from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from src.utils.file_utils import save_json


# ================================== call_back classes ==================================
class Seq2SeqLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)
        # pl_module.logger.log_metrics(trainer.callback_metrics, step=trainer.global_step)  # TODO: uncomment it

    @rank_zero_only
    def _write_logs(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, file_prefix: str, save_generations=True
    ) -> None:
        print(f"***** {file_prefix} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        output_dir = Path(pl_module.experiment_output_dir)
        if file_prefix == "test":
            results_file = output_dir / "test_results.txt"
            generations_file = output_dir / "test_generations.txt"
        else:
            results_file = output_dir / f"{file_prefix}_results/{trainer.global_step:05d}.txt"
            generations_file = output_dir / f"{file_prefix}_generations/{trainer.global_step:05d}.txt"

        results_file.parent.mkdir(exist_ok=True)
        generations_file.parent.mkdir(exist_ok=True)

        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            nparams = pl_module.model.model.num_parameters()
        except AttributeError:
            nparams = pl_module.model.num_parameters()

        # mp stands for million parameters
        params_stat = {"n_params": nparams, "mp": nparams / 1e6}
        trainer.logger.log_metrics(params_stat)
        print(f"Training is started! params statistics: {params_stat}")

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)


class LoggingCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        pl_module.logger.log_metrics(metrics)
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


class Seq2SeqCheckpointCallback(pl.callbacks.ModelCheckpoint):
    available_metrics = ["val_rouge2", "val_bleu", "val_loss"]

    def __init__(self, output_dir, experiment_name, monitor="val_loss",
                 save_top_k=1, every_n_train_steps=1000, verbose=False, **kwargs):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.every_n_train_steps = every_n_train_steps
        self.verbose = verbose
        self.check_monitor_validity(self.monitor)
        super(Seq2SeqCheckpointCallback, self).__init__(dirpath=self.output_dir,
                                                        filename=f"{self.experiment_name}" +
                                                                 '-{epoch:02d}-{step}-{' +
                                                                 f"{self.monitor}" + ':.4f}',
                                                        auto_insert_metric_name=True,
                                                        every_n_train_steps=self.every_n_train_steps,
                                                        verbose=self.verbose,
                                                        monitor=self.monitor,
                                                        mode="min",
                                                        save_top_k=self.save_top_k,
                                                        **kwargs)

    def check_monitor_validity(self, monitor):
        """Saves the best model by validation ROUGE2 score."""
        if monitor in self.available_metrics:
            pass
        else:
            raise NotImplementedError(
                f"seq2seq callbacks only support {self.available_metrics}, got {monitor}, "
                f"You can make your own by adding to this function."
            )


class SaveCheckpointEveryEpoch(pl.callbacks.ModelCheckpoint):
    """save checkpoint after each training epoch without validation.
    if ``last_k == -1``, all models are saved. and no monitor needed in this condition.
    otherwise, please log ``global_step`` in the training_step. e.g. self.log('global_step', self.global_step)

    :param last_k: the latest k models will be saved.
    :param save_weights_only: if ``True``, only the model's weights will be saved,
    else the full model is saved.
    """

    def __init__(self, last_k, save_weights_only, output_dir, experiment_name, **kwargs):
        if last_k == -1:
            super().__init__(save_top_k=-1, save_last=False, save_weights_only=save_weights_only, dirpath=output_dir,
                             filename=f"{experiment_name}" + '-{epoch:02d}-{step}', **kwargs)  # TODO add {val_loss}
        else:
            super().__init__(monitor='step', mode='max', save_top_k=last_k,
                             save_last=False, save_weights_only=save_weights_only, dirpath=output_dir,
                             filename=f"{experiment_name}" + '-{epoch:02d}-{step}',
                             save_on_train_epoch_end=True, **kwargs)
