"""
@Desc:
@Reference:
- transformers examples for using BART model
https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization
https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/2
- add_special_tokens
https://huggingface.co/docs/transformers/v4.17.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
- linecache
https://blog.csdn.net/my2010Sam/article/details/38022041
- torch Dataset
https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
@Notes:
- add_special_tokens
special_tokens_dict (dictionary str to str or tokenizers.AddedToken) —
Keys should be in the list of predefined special attributes:
[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
Tokens are only added if they are not already in the vocabulary (tested by checking
if the tokenizer assign the index of the unk_token to them).
- collate_fn
A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch.
See this section on more about collate_fn.
"""

import os
import copy
import transformers
import torch

from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import Dataset
from datasets import load_dataset
from src.utils.print_utils import print_rank_0

IGNORE_INDEX = -100


class PretrainingDataset(Dataset):
    def __init__(self, config, tokenizer, data_type):
        self.config = config
        self.tokenizer = tokenizer
        self.data = load_dataset('json', data_files=os.path.join(config.data_dir, f'{data_type}.json'))['train']
        self.print_dataset_example(example=self[0])

    def collate_fn(self, batch):
        input_ids = torch.tensor([sample["input_ids"] for sample in batch])
        return {"input_ids": input_ids, "labels": input_ids}

    def print_dataset_example(self, example):
        print("input_ids", example["input_ids"])
        print("inputs", self.tokenizer.decode(example["input_ids"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data[index]['token']
        return {'input_ids': input_ids}


class SupervisedDataset(Dataset):
    def __init__(self, config, tokenizer, data_type):
        self.config = config
        self.tokenizer = tokenizer
        self.data = load_dataset('json', data_files=os.path.join(config.data_dir, f'{data_type}.json'))['train']
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        print_rank_0(self.data[0])

    def _add_speaker_and_signal(self, conversation):
        meta_instruction = ("一位用户和法律大模型韩非之间的对话。"
                            "对于用户的法律咨询，韩非给出准确的、详细的、温暖的指导建议。"
                            "对于用户的指令问题，韩非给出有益的、详细的、有礼貌的回答。\n\n")
        conversation_roles = ("用户", "韩非")

        def tokenize(prompt):
            result = self.tokenizer(prompt, max_length=self.config.max_source_length,
                                    truncation=True, padding=False)
            return {"input_ids": result["input_ids"], "labels": copy.deepcopy(result["input_ids"])}

        user_sep, sys_sep = " ", self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        input_ids = tokenize(meta_instruction + user_sep)['input_ids']  # NOTE: + user_sep
        labels = [IGNORE_INDEX] * len(input_ids)
        for turn in conversation:
            if turn["from"].lower() == "gpt":
                role = conversation_roles[1]
                sent = tokenize(role + "：" + turn["value"] + (sys_sep if turn["value"] else ""))
                input_ids += sent['input_ids']
                labels += sent['labels']
            else:
                role = conversation_roles[0]
                sent = tokenize(role + "：" + turn["value"] + (user_sep if turn["value"] else ""))
                input_ids += sent['input_ids']
                labels += [IGNORE_INDEX] * len(sent['labels'])
        input_ids = torch.tensor(input_ids[:self.config.max_source_length])
        labels = torch.tensor(labels[:self.config.max_source_length])
        return input_ids, labels

    def conversation_data(self, data_point):
        conversation = data_point['conversations']
        input_ids, labels = self._add_speaker_and_signal(conversation)
        return {'input_ids': input_ids, 'labels': labels}

    def preprocess(self, data_point):
        return self.conversation_data(data_point)

    def collate_fn(self, batch):
        processed_batch = [self.preprocess(x) for x in batch]
        return self.data_collator(processed_batch)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

