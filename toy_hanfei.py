from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer


print("toy starting")

model_path = "/HanFei/output/task/bloomz-7b1-mt-gpu8-1e5/models/global_step26308"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

input_text = '中国人民共和国刑法第一条是：'
print(input_text)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print(generator(input_text, max_length=1024, min_length=256, num_return_sequences=1))
