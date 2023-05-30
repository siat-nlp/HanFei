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
os.environ["TOKENIZERS_PARALLELISM"] = "false" because
 https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
"""

import os
import argparse

import pytorch_lightning as pl
from src.utils.string_utils import str2bool

from src.configuration.constants import BASE_DIR
from src.configuration.pl_argsparser import (
    set_basic_args_for_pl_trainer,
    set_speedup_args_for_pl_trainer,
    set_device_args_for_pl_trainer,
    process_parsed_args_for_pl_trainer,
)

EXPERIMENT_GROUP = "task"
MODEL_NAME = "know_pre"
OUTPUT_DIR = f'{BASE_DIR}/output/{EXPERIMENT_GROUP}'
DATASETS_DIR = f'{BASE_DIR}/resources'
RESOURCES_DIR = f'{BASE_DIR}/resources'
DATA_NAME = "roc-stories"
MODEL_NAME_OR_PATH = f'{RESOURCES_DIR}/external_models/t5-base'


def add_customized_args(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_source_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_embeds", action="store_true")
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--logger_name", type=str, choices=["CSVLogger", "WandbLogger"], default="CSVLogger")
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--eval_beams", type=int, default=4, required=False)
    parser.add_argument("--num_beam_groups", type=int, default=0, required=False)
    parser.add_argument("--repetition_penalty", type=int, default=0, required=False)
    parser.add_argument("--num_return_sequences", type=int, default=20, required=False)
    parser.add_argument("--diversity_penalty", type=int, default=0, required=False)
    parser.add_argument("--r_drop_alpha", type=int, default=5, required=False)
    parser.add_argument(
        "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        required=False,
        help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. "
             "So val_check_interval will effect it.",
    )
    parser.add_argument("--fast_generate", action="store_true", default=False,
                        help="for _generative_step")
    parser.add_argument("--remain_sp_tokens", action="store_true", default=False,
                        help="remain special tokens in target and pred text (e.g. [EVENT_s])")
    parser.add_argument("--training_stage", type=str, choices=["pre_training", "instruction_tuning"],
                        default="pre_training", help="Training stage")

    return parser


def add_args_for_pytorch_lightning(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--encoder_layerdrop",
        type=float,
        help="Encoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--decoder_layerdrop",
        type=float,
        help="Decoder layer dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        help="Attention dropout probability (Optional). Goes into model.config",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        type=str,
        help="Learning rate scheduler",
    )
    parser.add_argument("--weight_decay", default=0., type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--optimizer_class", type=str, default="AdamW", help="optimizers: Adafactor|AdamW")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('PT_OUTPUT_DIR', OUTPUT_DIR),
                        help='directory to save logs and checkpoints to')
    parser.add_argument('--data_dir', type=str,
                        default=f'{DATASETS_DIR}',
                        help='directory with train, dev, test files')
    parser.add_argument('--index_ratio', type=int,
                        default=2,
                        help='multi-task ratio')
    parser.add_argument('--qa_exist', type=int, choices=[0, 1],
                        default=1,
                        help='train on qa task')

    parser.add_argument('--id_len', type=int,
                        default=8,
                        help='length of id')
    parser.add_argument('--resources_dir', type=str,
                        default=f'{RESOURCES_DIR}',
                        help='directory with of resources, including pretrained off-line models.')
    parser.add_argument("--experiment_name", default=f"{MODEL_NAME}-{DATA_NAME}",
                        type=str, help="the name of the experiment.")
    parser.add_argument("--model_name", default=f"{MODEL_NAME}",
                        type=str, help="the name of the model used.")
    parser.add_argument("--model_name_or_path",
                        default=MODEL_NAME_OR_PATH,
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate.")
    parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_val", type=int, default=1000, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True, help="overwrite_output_dir")
    parser.add_argument("--train_batch_size", default=10, type=int, help="train_batch_size.")
    parser.add_argument("--eval_batch_size", default=10, type=int, help="eval_batch_size.")

    # ############################### checkpoint settings ####################################
    parser.add_argument("--save_top_k", default=-1, type=int,
                        help="The best k models according to the quantity monitored will be saved.")
    parser.add_argument("--save_every_n_epochs", default=None, type=int,
                        help="Save on every n epochs")
    parser.add_argument("--save_every_n_steps", default=None, type=int,
                        help="Save on every n steps")
    parser.add_argument("--every_n_val_epochs", default=1, type=int,
                        help="every_n_val_epochs.")
    parser.add_argument("--ckpt_verbose", type=str2bool, default=False,
                        help="verbosity mode. True / False")

    return parser


def parse_args_for_config(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)  # Extends existing argparse by default attributes for cls.
    # set defaults for args of pl.trainer
    parser = set_basic_args_for_pl_trainer(parser, output_dir=OUTPUT_DIR)
    parser = set_speedup_args_for_pl_trainer(parser, amp_backend="native", precision=16)
    parser = set_device_args_for_pl_trainer(parser)

    # customized
    parser = add_args_for_pytorch_lightning(parser)
    parser = add_customized_args(parser)
    args = parser.parse_args()

    # process parsed args
    process_parsed_args_for_pl_trainer(args)
    return args


if __name__ == '__main__':
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args_for_config()
