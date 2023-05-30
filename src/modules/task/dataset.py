import json
import torch
import transformers

from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
from typing import Dict
from conversation import get_default_conv_template, SeparatorStyle
from src.utils.print_utils import print_rank_0


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.TWO

    # TODO Mask targets
    # sep = conv.sep + conv.roles[1] + ": "
    # for conversation, target in zip(conversations, targets):
    #     total_len = int(target.ne(tokenizer.pad_token_id).sum())
    #
    #     rounds = conversation.split(conv.sep2)
    #     cur_len = 1
    #     target[:cur_len] = IGNORE_TOKEN_ID
    #     for i, rou in enumerate(rounds):
    #         if rou == "":
    #             break
    #
    #         parts = rou.split(sep)
    #         if len(parts) != 2:
    #             break
    #         parts[0] += sep
    #         round_len = len(tokenizer(rou).input_ids)
    #         instruction_len = len(tokenizer(parts[0]).input_ids) - 2
    #
    #         target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
    #
    #         cur_len += round_len
    #     target[cur_len:] = IGNORE_TOKEN_ID
    #
    #     if False:
    #         z = target.clone()
    #         z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
    #         rank0_print(tokenizer.decode(z))
    #
    #     if cur_len < tokenizer.model_max_length:
    #         if cur_len != total_len:
    #             target[:] = IGNORE_TOKEN_ID
    #             rank0_print(
    #                 f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
    #                 f" (ignored)"
    #             )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        print_rank_0("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print_rank_0("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, lazy_preprocess) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset if lazy_preprocess else SupervisedDataset)
    print_rank_0("Loading data...")
    raw_data = json.load(open(data_path, "r"))
    print_rank_0(f"#data {len(raw_data)}")
    dataset = dataset_cls(raw_data, tokenizer=tokenizer)
    return dict(dataset=dataset)
