"""
@Desc:
@Reference:
@Notes:
WANDB is Weights and Biases Logger:
https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
"""

import sys
import json
import numpy as np
from pathlib import Path

from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.task.config_args import parse_args_for_config
from src.utils.file_utils import copy_file_or_dir, output_obj_to_file, pickle_save, pickle_load, joblib_save, \
    joblib_load
from src.utils import nlg_eval_utils
from train import EventTriggerTrainer


class EventTriggerTester(EventTriggerTrainer):
    def __init__(self, args):
        # parameters
        super().__init__(args)
        self.generation_dir = self.experiment_output_dir / "gen_result"
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = self.model.tokenizer
        self.model.eval()
        self.test_output = None
        self.src_file = None
        self.tgt_file = None
        self.gen_file = None
        self.eval_file = None

        # customized
        self.dataset = self.model.test_dataloader().dataset
        self.output_prefix = f"{self.model.model_name}"
        self.test_output_store_path = self.cache_dir.joinpath(f"{self.output_prefix}_test_output.pkl")
        self.ppl_file = self.generation_dir / f"{self.output_prefix}_ppl.job"
        self.acc_file = self.generation_dir / f"{self.output_prefix}_acc.job"
        self.freq_acc_file = self.generation_dir / f"{self.output_prefix}_freq_acc.job"

    def test(self, ckpt_path=None):
        # if ckpt_path is None:
        #     ckpt_path = self.checkpoints[-1]
        self.pl_trainer.test(model=self.model, ckpt_path=None)

    def init_test_output(self):
        if self.test_output_store_path.exists():
            print(f"test output loaded from {self.test_output_store_path}")
            self.test_output = pickle_load(self.test_output_store_path)
        if self.test_output is None:
            self.model.store_test_output = True
            self.test()
            self.test_output = self.model.test_output
            print(f"test output stored to {self.test_output_store_path}")
            pickle_save(self.test_output, self.test_output_store_path)
        if self.test_output is None:
            raise ValueError("self.test_output cannot be None")

    def generate(self):
        self.init_test_output()
        print(f"model {self.model.model_name} generating")
        print(f"src_file: {self.src_file}\ntgt_file: {self.tgt_file}\ngen_file: {self.gen_file}\n")
        print(f"test_loss: {self.test_output['test_loss']}")
        # print(f"metrics: {self.test_output['log']}")

        joblib_save(self.test_output['log']['test_logits_acc'], self.acc_file)
        joblib_save(self.test_output['log']['test_logits_ppl'], self.ppl_file)
        print(f"test_logits_acc saved at {self.acc_file}")
        print(f"test_logits_ppl saved at {self.ppl_file}")

    # def eval_output(self):
    #     self.init_test_output()
    #     test_logits_acc = self.test_output['log']['test_logits_acc']
    #     test_logits_ppl = self.test_output['log']['test_logits_ppl']
    #
    #     print("=" * 10)
    #
    #     print(f"model {self.model.model_name} eval {self.gen_file}")
    #     output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)
    #     return metrics


if __name__ == '__main__':
    hparams = parse_args_for_config()
    tester = EventTriggerTester(hparams)

    # generate predicted stories
    tester.generate()
    # tester.eval_output()
