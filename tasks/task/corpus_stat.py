import sys
import torch
import random
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # 在tasks文件夹中可以直接运行程序

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from src.utils.file_utils import pickle_save, pickle_load
from tqdm import tqdm

data = load_dataset('NeelNanda/c4-code-tokenized-2b', cache_dir='resources/')['train']

neox_digits_tokenizer = PreTrainedTokenizerFast.from_pretrained('NeelNanda/gpt-neox-tokenizer-digits',
                                                                cache_dir='resources/')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('GPT2-large', cache_dir='cache_dir/')
token_freq = {}
for t in range(gpt2_tokenizer.vocab_size):
    token_freq[t] = 0
count = 0
for piece in tqdm(data, total=100000):
    if count == 100000:
        break
    piece_sentence = neox_digits_tokenizer.decode(piece['tokens'], skip_special_tokens=True)
    piece_gpt2 = gpt2_tokenizer(piece_sentence)['input_ids']
    for token in piece_gpt2:
        token_freq[token] += 1
    count += 1

token_freq = sorted(token_freq.items(), key=lambda kv: (kv[1], kv[0]))
bucket_length = len(token_freq) // 10
count = 0
bucket_num = 0
token_to_freq = {}
for token in token_freq:
    if count > bucket_length:
        count = 0
        bucket_num += 1
    token_to_freq[token] = bucket_num
    count += 1
pickle_save(token_to_freq, 'resources/token_to_freq.pkl')
print('token to freq saved')
