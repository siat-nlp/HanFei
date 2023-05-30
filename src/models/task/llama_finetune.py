"""
@Desc:
@Reference:
- BART shift_tokens_right
https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/bart#bart
- Label Smoothing
https://paperswithcode.com/method/label-smoothing
- bart models from huggingface
e.g. https://huggingface.co/facebook/bart-base
@Notes:
- BART shift_tokens_right
Bart uses the eos_token_id as the starting token for decoder_input_ids generation.
If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values).
For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided,
the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
- label-smoothing
During finetuning we use a label smoothed cross entropy loss (Pereyra et al., 2017), with the smoothing parameter
set to 0.1.
- model generate:
in generation_utils.py e.g.BartForConditionalGeneration().generate -> def generate in generation_utils.py
- torch.nn.CrossEntropyLoss
    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`(N)` where each value is
          :math:`0 \leq \text{targets}[i] \leq C-1`, or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss. If containing class probabilities,
          same shape as the input.
"""

import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers.models.gpt_neo import modeling_gpt_neo
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoConfig

from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import GPT2Tokenizer
from torch.nn import CrossEntropyLoss

from src.utils.gen_utils import ids_to_clean_string, top_p_logits
from src.utils import nlg_eval_utils
from src.utils.nlg_eval_utils import metric_max_over_ground_truths, exact_match_score
from src.modules.task.data_loader import PretrainingDataset
from src.utils.task import model_utils

from src.models.lightning_base import BaseTransformer

logger = logging.getLogger(__name__)


class llama_finetune(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams,
                         **kwargs)

        self._custom_init()

        # Whether changing embeddings
        if self.hparams.freeze_embeds:
            model_utils.freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            model_utils.freeze_params(self.model.get_encoder())
            model_utils.assert_all_frozen(self.model.get_encoder())

        self.step_count = 0
        self.current_val_metrics = {}
        self.metrics_save_path = Path(self.experiment_output_dir) / "metrics.json"
        self.metrics: dict = defaultdict(list)
        self.model_type = self.config.model_type
        self.decoder_start_token_id = self.model.config.decoder_start_token_id  # default to config
        self.already_saved_batch = False  # flag of saving readable batch
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        self.max_output_length = self.hparams.max_target_length
        self.val_metric = "loss" if self.hparams.val_metric is None else self.hparams.val_metric
        self.save_readable_batch = True  # for debug
        self.metric_names_update_flag = True

        # predicted
        self.use_top_p = False
        self.top_p = 0.9
        self.store_test_output = True
        self.test_output = None
        self.remain_sp_tokens = self.hparams.remain_sp_tokens
        from src.utils.file_utils import pickle_load
        # self.token_to_freq = pickle_load("resources/token_to_freq.pkl")
        if self.remain_sp_tokens:
            print("remain special tokens in target and pred text (e.g. [EVENT_s])")

    def _custom_init(self):
        # load pretrained settings from bart
        # config
        from src.flash_bloom import llama_flash_attn_monkey_patch
        llama_flash_attn_monkey_patch.replace_llama_attn_with_flash_attn()

        self.config: AutoConfig = AutoConfig.from_pretrained(self.hparams.model_name_or_path, trust_remote_code=True)
        # tokenizer
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path,
                                                                      trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # model
        self.model = self._load_model(self.hparams.model_name_or_path, AutoModelForCausalLM, self.config)
        from src.flash_bloom.flash_attn_wrapper import FlashAttentionWrapperWithAlibi
        for each in self.model.transformer.h:
            each.self_attention = FlashAttentionWrapperWithAlibi(each.self_attention, max_seqlen=512)
        self._set_up(config=self.config,
                     tokenizer=self.tokenizer,
                     model=self.model)
        IGNORE_INDEX = -100
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "</s>"
        DEFAULT_UNK_TOKEN = "</s>"
        model_utils.smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

        self.train_dataset_class = gpt_Dataset_dialogue
        self.test_dataset_class = gpt_Dataset_dialogue

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def gather_nd(self, x, indices):
        newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
        indices = indices.view(-1, indices.shape[-1]).tolist()
        out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
        return out

    def _step(self, batch: dict):
        outputs = self.model(**batch, use_cache=False)

        return outputs

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)['loss']
        logs = {"train_loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(logs, sync_dist=True, on_step=True)
        # logs["batch_size"] = batch["input_ids"].shape[0]
        return {"loss": loss, "log": logs}

    def gen_ids_to_clean_text(self, generated_ids: List[int]):
        gen_list = []
        for output in generated_ids:
            # gen_list.append(ids_to_clean_string(output, self.tokenizer, remain_sp_tokens=self.remain_sp_tokens))
            gen_list.append(self.tokenizer.decode(output, skip_special_tokens=True))
        return gen_list

    def filter_qa(self, preds: List[str]):
        filter_preds = []
        for output in preds:
            if len(output) > self.hparams.id_len:
                filter_preds.append(output[self.hparams.id_len:])
            else:
                filter_preds.append(output)
        return filter_preds

    @torch.no_grad()
    def _generative_step(self, batch: dict) -> dict:
        tik = datetime.now()
        extra_params = {}
        if self.hparams.num_beam_groups > 1:
            extra_params["num_beam_groups"] = self.hparams.num_beam_groups
            extra_params["diversity_penalty"] = self.hparams.diversity_penalty
        if self.eval_beams >= 1:
            extra_params["num_beams"] = self.eval_beams
        if self.hparams.repetition_penalty > 0:
            extra_params["repetition_penalty"] = self.hparams.repetition_penalty
        if self.hparams.num_return_sequences > 0:
            extra_params["num_return_sequences"] = self.hparams.num_return_sequences
            extra_params["num_beams"] = self.hparams.num_return_sequences
        outputs = self._step(batch)
        loss = outputs['loss']
        logits = outputs['logits']

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        logits_50 = shift_logits[..., [50], :]
        logits_500 = shift_logits[..., [500], :]
        labels_50 = shift_labels[..., [50]]
        labels_500 = shift_labels[..., [500]]
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        icl_loss_50 = loss_fct(logits_50.view(-1, logits_50.size(-1)), labels_50.view(-1))
        icl_loss_500 = loss_fct(logits_500.view(-1, logits_500.size(-1)), labels_500.view(-1))
        icl_loss = icl_loss_50 - icl_loss_500

        tok = datetime.now()
        batch_gen_time = tok - tik
        base_metrics = {"loss": loss.item(), "icl_loss": icl_loss.item()}

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time)

        return base_metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> Dict:
        loss = self._step(batch)['loss']
        logs = {"val_loss": loss.item()}
        # metrics logged can be access by trainer.callback_metrics
        self.log_dict(self.current_val_metrics, sync_dist=True)
        logs["batch_size"] = batch["input_ids"].shape[0]
        return {"loss": loss, "log": logs}

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        generative_metrics = {
            name: np.array(torch.tensor([x[name] for x in outputs]).cpu()).mean() for name in self.metric_names
        }
        metric_val = (
            torch.tensor(generative_metrics[self.val_metric])
        )
        val_metrics = {f"{prefix}_{k}": x for k, x in generative_metrics.items()}
        val_metrics["step_count"] = float(self.step_count)
        self.current_val_metrics = val_metrics
        self.metrics[prefix].append(val_metrics)  # callback writes this to self.metrics_save_path.
        print(f"Evaluation result: {val_metrics}")
        return {
            "log": val_metrics,
            f"{prefix}_loss": generative_metrics["loss"],
            f"{prefix}_{self.val_metric}": metric_val,
        }

    def test_step(self, batch, batch_idx):
        tik = datetime.now()
        outputs = self._step(batch)
        loss = outputs['loss']
        logits = outputs['logits']

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()

        shift_logits_argmax = torch.argmax(shift_logits, dim=-1)
        logits_acc = (shift_logits_argmax == shift_labels).byte()

        logits_ppl = []
        for log, lab in zip(shift_logits, shift_labels):
            logits_ppl.append(-torch.nn.functional.log_softmax(log, dim=-1).gather(1, lab.unsqueeze(-1)).squeeze(1))

        logits_ppl = torch.stack(logits_ppl)

        tok = datetime.now()
        batch_gen_time = tok - tik
        base_metrics = {"loss": loss.item(), "logits_acc": logits_acc.cpu(), "logits_ppl": logits_ppl.cpu(),
                        "shift_logits_argmax": shift_logits_argmax.cpu(), "shift_labels": shift_labels.cpu()}

        # update metric_names
        self.update_metric_names(base_metrics, update_flag=self.metric_names_update_flag)
        self.metric_names_update_flag = False
        base_metrics.update(batch_gen_time=batch_gen_time)

        return base_metrics

    def test_epoch_end(self, outputs):
        prefix = 'test'
        self.step_count += 1
        loss_metrics = {}

        for name in self.metric_names:
            if 'loss' in name:
                loss_metrics[name] = np.array([x[name] for x in outputs]).mean()
            else:
                loss_metrics[name] = [x[name].numpy().tolist() for x in outputs]

        metric_val = (
            torch.tensor(loss_metrics[self.val_metric])
        )

        val_metrics = {f"{prefix}_{k}": x for k, x in loss_metrics.items()}
        val_metrics["step_count"] = float(self.step_count)
        self.current_val_metrics = val_metrics
        self.metrics[prefix].append(val_metrics)  # callback writes this to self.metrics_save_path.
        # print(f"Predict result: {val_metrics}")
        test_output = {
            "log": val_metrics,
            f"{prefix}_loss": loss_metrics["loss"],
            f"{prefix}_{self.val_metric}": metric_val,
        }
        if self.store_test_output:
            self.test_output = test_output
        return test_output

    def get_dataset(self, data_type):
        if data_type == 'train':
            dataset = self.train_dataset_class(self.hparams, self.tokenizer, 'train')
        else:
            dataset = self.test_dataset_class(self.hparams, self.tokenizer, 'test')
        # self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
        return dataset

    def get_dataloader(self, data_type: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(data_type)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        train_shuffle = True if self.hparams.overfit_batches == 0.0 else False
        if not train_shuffle:
            print(f"train_shuffle: {train_shuffle} overfit_batches: {self.hparams.overfit_batches}")
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size,
                                   shuffle=train_shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size,
                                   shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size,
                                   shuffle=True)
