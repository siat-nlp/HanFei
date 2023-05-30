from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
import ast
from collections import Counter
import json
from statistics import mode
from tqdm import tqdm


def prompt_without_equtation(dataset):
    question = dataset["question"]
    answer = list(map(lambda x: re.sub("<<.*>>", "", x), dataset["answer"]))
    answer = list(map(lambda x: re.sub("\n####", " The answer is:", x), answer))
    input_list = []
    prompt = []
    for q in range(len(question)):
        prompt.append("\nQ: " + question[q] + "\nA: " + answer[q] + " \n")
    return prompt


def add_prompts_to_input(sample_from_prompts, test_question):
    test_input_with_prompts = sample_from_prompts + "\nQ: " + test_question + "\nA: "
    return test_input_with_prompts


def prompt_with_equtation(dataset):
    question = dataset["question"]
    answer = list(
        map(lambda x: re.sub("\n####", " The answer is:", x), dataset["answer"])
    )
    input_list = []
    prompt = []
    for q in range(len(question)):
        prompt.append("\nQ: " + question[q] + "\nA: " + answer[q] + " \n")
    return prompt


MODEL_NAME = "/mntnfs/med_data5/zhanghongbo/general_pretrain/output/task/GPT_Code/best_tfmr"
# MODEL_NAME = "gpt2-large"
dataset = load_dataset("gsm8k", 'main', cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                          cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 50256
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
model = model.cuda()

input_prompts = prompt_without_equtation(dataset["train"])
random.seed(42)
number_of_prompts = 4
sample_from_prompts = " ".join(
    map(str, random.sample(input_prompts, number_of_prompts))
)

test_same_prompts = list(
    map(
        lambda x: add_prompts_to_input(sample_from_prompts, x),
        dataset["test"]["question"],
    )
)
print(sample_from_prompts)

input_prompts_with_equtation = prompt_with_equtation(dataset["train"])
random.seed(42)
sample_from_prompts_with_equtation = " ".join(
    map(str, random.sample(input_prompts_with_equtation, number_of_prompts))
)
test_same_prompts_with_equtation = list(
    map(
        lambda x: add_prompts_to_input(sample_from_prompts_with_equtation, x),
        dataset["test"]["question"],
    )
)

# test_same_prompts = test_same_prompts[:20]
# test_same_prompts_with_equtation = test_same_prompts_with_equtation[:20]

results_greedy = []
for i in [
    test_same_prompts,
    test_same_prompts_with_equtation,
]:
    predicted_data = []
    for j in tqdm(i):
        inputs = tokenizer(j, return_tensors="pt")["input_ids"].cuda()
        outputs = model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        predicted_data.append(tokenizer.decode(outputs[0]))
    results_greedy.append(predicted_data)

with open("results_greedy", "w") as fp:
    json.dump(results_greedy, fp)


def get_answer_from_generated_text(generated_text, question_from_test):
    answer = str()
    result = re.search(
        r"The answer is: \d+", generated_text.split(question_from_test)[-1]
    )
    try:
        answer = re.search(r"\d+", result.group(0)).group(0)
    except:
        answer = "UNK"
    return answer


test_dataset = dataset["test"]["question"]
answers_test = []
for i in dataset["test"]["answer"]:
    answers_test.append(i.split("\n#### ")[-1])

all_answers = []
for dataset in results_greedy:
    results_chain_of_thoughts = []
    for example in tqdm(range(len(dataset))):
        results_chain_of_thoughts.append(
            get_answer_from_generated_text(dataset[example], test_dataset[example])
        )
    all_answers.append(results_chain_of_thoughts)

results_total_chain_of_thoughts = []
for answers in tqdm(range(len(all_answers))):
    results_by_method = []
    for generated_answer in range(len(all_answers[answers])):
        if all_answers[answers][generated_answer] == answers_test[generated_answer]:
            results_by_method.append("yes")
        else:
            results_by_method.append("no")
    results_total_chain_of_thoughts.append(results_by_method)

for i in tqdm(results_total_chain_of_thoughts):
    precision = int(Counter(i)["yes"] / len(i) * 100)
    print(f"Precision: {precision}%")


def get_maj_answer(answers):
    while "UNK" in answers:
        answers.remove("UNK")
    if answers:
        mode_answer = mode(answers)
    else:
        mode_answer = "UNK"
    return mode_answer


# Посчитаем совпадения в ответах на тесте и в сгенерированных примерах. Yes – ответ совпал, no – не совпал.
# UNK в данном случае также относиться к no, так как ответа на выходе мы не получаем
results_total_self_consistency = []
for answers in tqdm(range(len(all_answers_self_consistency))):
    results_by_method = []
    for generated_answer in range(len(all_answers_self_consistency[answers])):
        if (
                all_answers_self_consistency[answers][generated_answer]
                == answers_test[generated_answer]
        ):
            results_by_method.append("yes")
        else:
            results_by_method.append("no")
    results_total_self_consistency.append(results_by_method)

for i in tqdm(results_total_self_consistency):
    precision = int(Counter(i)["yes"] / len(i) * 100)
    print(f"Precision: {precision}%")
