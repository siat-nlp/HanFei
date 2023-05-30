import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy


# openai.api_key = "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"

# openai.api_key = "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"  # wo

openai.api_key = "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"  # qin


# model_list = openai.Model.list()

# datas = model_list['data']

# models = [data['id'] for data in datas]

# print(models)


StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

# openai_org = os.getenv("OPENAI_ORG")
# if openai_org is not None:
#     openai.organization = openai_org
#     logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    model : str = "gpt-3.5-turbo"
    # max_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    # stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # suffix: Optional[str] = None
    # logprobs: Optional[int] = None
    # echo: bool = False
    logit_bias = {"50256": -100}
    stop=["\n[6].", "[6]."]

def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:

    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for prompt_batch in prompt_batches:
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args
        while True:
            try:
                shared_kwargs = dict(
                    **batch_decoding_args.__dict__,
                )
                # print(shared_kwargs)
                messages = [
                    {"role":"system","content": "你是一个法律助手。"},
                    {"role":"user","content": prompt_batch[0]}
                ]
                completion_batch = openai.ChatCompletion.create(messages=messages, **shared_kwargs)

                # print(prompt_batch)
                # print(len(prompt_batch[0]))
                # completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices
                # text = choices[0].text
                # print(text)
                # print(len(text))
                # completions.append(text)
                for choice in choices:
                    # print(choice.message.content)
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                print(e)
                # print(len(prompt_batch[0]))
                # max_len = int(max_len * 0.9)
                # logging.warning(f"OpenAIError: {e}.")
                # if "Please reduce your prompt" in str(e):
                #     batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.9)
                #     logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                # else:
                #     logging.warning("Hit request rate limit; retrying...")
                #     time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="a", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
