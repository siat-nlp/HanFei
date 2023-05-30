"""
@Desc:
@Reference:
- pytorch_lightning documentation about trainer parameters:
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html?
highlight=lighting_logs#pytorch_lightning.trainer.Trainer.params.flush_logs_every_n_steps
- pl.Trainer.add_argparse_args:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.utilities.argparse.html
?highlight=add_argparse_args#pytorch_lightning.utilities.argparse.add_argparse_args
@Notes:
parser.set_defaults(x="...") don't require parser.add_argument("--x", ...)
"""

import argparse
import torch


def set_device_args_for_pl_trainer(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    if torch.cuda.is_available():
        # use gpu
        # If your machine has GPUs, it will use the GPU Accelerator for training
        parser.set_defaults(accelerator="gpu")
        # Number of GPUs to train on (int)
        parser.set_defaults(gpus=torch.cuda.device_count())
    else:
        # use cpu
        parser.set_defaults(accelerator="cpu")
        parser.set_defaults(devices=1)
        parser.set_defaults(gpus=0)
    return parser


def set_basic_args_for_pl_trainer(parser: argparse.ArgumentParser = None, output_dir=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    if output_dir is None:
        raise ValueError("Must input the output_dir")
    # "Number of updates steps to accumulate before performing a backward/update pass."
    parser.add_argument('--accum_batches_args', type=str, help='set accumulate_grad_batches')
    # Stop training once this number of epochs is reached
    parser.set_defaults(max_epochs=10)
    # Force training for at least these many epochs
    # parser.set_defaults(min_epochs=2)
    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
    parser.set_defaults(accumulate_grad_batches={5: 2, 10: 5})
    # Automatically tries to find the largest batch size that fits into memory, before any training.
    # Trainer(auto_scale_batch_size="binsearch") run batch size scaling, result overrides hparams.batch_size
    parser.set_defaults(auto_scale_batch_size=None)
    # default used by the Trainer (no learning rate finder)
    parser.set_defaults(auto_lr_find=False)
    # How often within one training epoch to check the validation set. Can specify as float or int.
    parser.set_defaults(val_check_interval=0.5)
    # Default path for logs and weights when no logger or pytorch_lightning.callbacks.ModelCheckpoint callback passed.
    parser.set_defaults(default_root_dir=output_dir)
    # default used by Trainer, saves the most recent model to a single checkpoint after each epoch
    parser.set_defaults(enable_checkpointing=True)
    # Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to
    # find any bugs (ie: a sort of unit test).
    parser.set_defaults(fast_dev_run=False)
    # How often to add logging rows (does not write to disk)
    parser.set_defaults(log_every_n_steps=50)
    # Sanity check runs n batches of val before starting the training routine.
    parser.set_defaults(num_sanity_val_steps=1)
    # Whether to enable or disable the progress bar. Defaults to True.
    parser.set_defaults(enable_progress_bar=True)
    # Enable synchronization between batchnorm layers across all GPUs.
    parser.set_defaults(sync_batchnorm=True)
    # Directory of where to save weights if specified.
    parser.set_defaults(weights_save_path=output_dir)
    # Gradient clipping value
    parser.set_defaults(gradient_clip_val=None)
    # Uses this much data of the training set. If nonzero, will turn off validation.
    # If the training dataloaders have shuffle=True, Lightning will automatically disable it.
    parser.set_defaults(overfit_batches=0.0)
    return parser


def set_speedup_args_for_pl_trainer(parser: argparse.ArgumentParser = None, amp_backend="apex", precision=16):
    if parser is None:
        parser = argparse.ArgumentParser()
    if amp_backend == "native":
        parser.set_defaults(precision=precision)
    elif amp_backend == "apex":
        # The optimization level to use (O1, O2, etc…) for 16-bit GPU precision (using NVIDIA apex under the hood).
        parser.set_defaults(amp_level='O2')
        # Lightning supports either double precision (64), full precision (32), or half precision (16) training.
        parser.set_defaults(precision=precision)
    else:
        raise NotImplementedError(f"amp_backend: {amp_backend}")
    # Use PyTorch AMP (‘native’) (available PyTorch 1.6+), or NVIDIA apex (‘apex’).
    parser.set_defaults(amp_backend=amp_backend)
    return parser


def process_parsed_args_for_pl_trainer(args: argparse.Namespace):
    if args.accum_batches_args is not None:
        batches = eval(args.accum_batches_args)
        print(f"reset accumulate_grad_batches to {batches}")
        args.accumulate_grad_batches = batches
    # precision
    if args.accelerator == "cpu" and args.precision == 16:
        args.precision = "bf16"
    else:  # gpu
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:  # multiple gpus
            args.accelerator = "gpu"
            # args.strategy = "ddp"
            args.strategy = "deepspeed_stage_3"
            # args.strategy = "fsdp"
            # args.precision = 32
