import torch
import argparse
import re
import json
import time
import os
import gradio as gr

from collections import namedtuple
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
tokenizer = None
model = None
LOAD_8BIT = False

ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            # scores[..., 20005] = 5e4
        return scores


def process_response(response):
    response = response.strip()
    if response[-4:] == "</s>":
        response = response[:-4]
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
        ["<\s>", ""],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)

    return response


def chat(model, tokenizer, query, history, max_length=2048, num_beams=1,
         do_sample=True, top_p=0.7, temperature=0.5, repetition_penalty=1.2, logits_processor=None, **kwargs):
    # if history is None:
    #     history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                  "temperature": temperature, "logits_processor": logits_processor,
                  "repetition_penalty": repetition_penalty, **kwargs}
    # if not history:
    #     prompt = query
    # else:
    #     prompt = ""
    #     for i, (old_query, response) in enumerate(history):
    #         prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
    #     prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    prompt = generate_prompt(query, history, tokenizer.eos_token)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)  # TODO support stream output
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = process_response(response)
    history = history + [[query, response]]
    if len(inputs["input_ids"][0]) > 1568:  # TODO improve this snippet to adapt max_length
        history = history[1:]
    return response, history


# def generate_prompt(query, history, eos):
#     if not history:
#         return f"""一个病人和人工智能医生HuatuoGPT之间的对话。HuatuoGPT对病人的问题给出有益的、详细的、有礼貌的回答。\n\n病人：{query} HuatuoGPT："""
#     else:
#         prompt = '一个病人和人工智能医生HuatuoGPT之间的对话。HuatuoGPT对病人的问题给出有益的、详细的、有礼貌的回答。\n\n'
#         for i, (old_query, response) in enumerate(history):
#             prompt += "病人：{} HuatuoGPT：{}".format(old_query, response) + eos
#         prompt += "\n病人：{} HuatuoGPT：".format(query)
#         return prompt

def generate_prompt(query, history, eos):
    if not history:
        return f"""一位用户和法律大模型韩非之间的对话。对于用户的法律咨询，韩非给出准确的、详细的、温暖的指导建议。对于用户的指令问题，韩非给出有益的、详细的、有礼貌的回答。\n\n 用户：{query} 韩非："""
        # return f"""A conversation between a patient and HuatuoGPT, an artificially intelligent doctor who gives helpful, detailed and polite answers to the patient's questions. \n\n Patient：{query} HuatuoGPT："""
    else:
        prompt = '一位用户和法律大模型韩非之间的对话。对于用户的法律咨询，韩非给出准确的、详细的、温暖的指导建议。对于用户的指令问题，韩非给出有益的、详细的、有礼貌的回答。\n\n '
        # prompt = "A conversation between a patient and HuatuoGPT, an artificially intelligent doctor who gives helpful, detailed and polite answers to the patient's questions. "
        for i, (old_query, response) in enumerate(history):
            prompt += "用户：{} 韩非：{}".format(old_query, response) + eos
        prompt += "用户：{} 韩非：".format(query)
        return prompt


def get_model_class_llama(model_path, cudaid):
    global llama_model, llama_tokenizer

    llama_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #   cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
    llama_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                       #  cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir',
                                                       ignore_mismatched_sizes=True).half().to(f'cuda:{cudaid}')
    llama_model = llama_model.eval()
    # model = None


def get_model_class_bloom(model_path, cudaid):
    global bloom_model, bloom_tokenizer
    kwargs = {"device_map": "auto", "max_memory": {0: "15GiB", 1: "15GiB"}}
    bloom_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #   cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
    bloom_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                       #  cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir',
                                                       ignore_mismatched_sizes=True, **kwargs)
    bloom_model = bloom_model.eval()


def get_model_class_Phoenix(model_path, cudaid):
    global Phoenix_model, Phoenix_tokenizer
    kwargs = {"device_map": "auto", "max_memory": {0: "15GiB", 1: "15GiB"}}
    Phoenix_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #   cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
    Phoenix_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                         #  cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir',
                                                         ignore_mismatched_sizes=True, **kwargs)
    Phoenix_model = Phoenix_model.eval()


def get_model_class_ChatGLM(model_path, cudaid):
    global glm_model, glm_tokenizer

    glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #   cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir')
    glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                          #  cache_dir='/mntnfs/med_data5/zhanghongbo/general_pretrain/cache_dir',
                                          ignore_mismatched_sizes=True).half().to(f'cuda:{cudaid}')
    glm_model = glm_model.eval()


def glm_predict(
        instruction,
        top_p=0.75,
        temperature=0.1,
        repetition_penalty=0.12,
        history=None,
        max_new_tokens=512,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    history = history or []
    response, history = glm_model.chat(glm_tokenizer, instruction, history=history, top_p=top_p,
                                       temperature=temperature,
                                       repetition_penalty=repetition_penalty)
    print(f"Assistant：{response}")
    return "", history, history


def llama_bloom_predict(
        instruction,
        top_p=0.75,
        temperature=0.5,
        repetition_penalty=0.12,
        history=None,
        max_new_tokens=512,
        top_k=40,
        num_beams=4,
        model=None,
        tokenizer=None,
        **kwargs,
):
    history = history or []
    response, history = chat(model, tokenizer, instruction, history, top_p=top_p, temperature=temperature,
                             repetition_penalty=repetition_penalty)
    print(f"Assistant：{response}")

    return "", history, history


def distribute(model_selector,
               instruction,
               top_p=0.75,
               temperature=0.1,
               repetition_penalty=1.2,
               history=None,
               max_new_tokens=512,
               top_k=40,
               num_beams=4,
               **kwargs):
    print(model_selector)
    if model_selector == 'HanFei':
        return llama_bloom_predict(instruction, top_p=top_p, temperature=temperature,
                                   repetition_penalty=repetition_penalty, history=history,
                                   model=bloom_model, tokenizer=bloom_tokenizer)
    else:
        print('wrong')


def predict_test(message, top_p, temperature, history):
    history = history or []

    user_message = f"{message} {top_p}, {temperature}"
    print(user_message)

    history.append((message, user_message))
    return history, history


def clear_session():
    return '', '', None


parser = argparse.ArgumentParser(description='Process some integers.')

args = parser.parse_args()

print(f'load model')
get_model_class_bloom('../../pretrain_model/hanfei-1.0', 1)
print(f'model had load')

block = gr.Blocks(
    css="""#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}""")

# disable_btn = gr.Button.update(interactive=False)
def upvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "upvote", model_selector, request)
    return ""


def downvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "downvote", model_selector, request)
    return ""


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open('vote_record.txt', "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def regeneratefunc(state):
    # dialogue_history = state["dialogue_history"]
    # if dialogue_history:
    #     dialogue_history.pop()

    # # 更新状态对象并保存到 Gradio 界面中
    # state["dialogue_history"] = dialogue_history
    # state.update()
    return (state[-1][0], state[:-1])


with block as demo:
    # top_p, temperature
    with gr.Accordion("Parameters", open=False):
        top_p = gr.Slider(minimum=-0, maximum=1.0, value=0.7, step=0.1, interactive=True,
                          label="Top-p (nucleus sampling)", )
        temperature = gr.Slider(minimum=-0, maximum=5.0, value=0.5, step=0.1, interactive=True, label="Temperature", )
        repetition_penalty = gr.Slider(minimum=1.0, maximum=5.0, value=1.2, step=0.1, interactive=True,
                                       label="repetition_penalty", )

    # models = ['HanFei', 'Huatuo200K-Bloom', 'Huatuo200K-Phoenix', 'Huatuo200K-Llama-half', 'Huatuo200K-ChatGLM-half']
    models = ['HanFei']
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False).style(container=False)

    chatbot = gr.Chatbot(label="HanFei")
    message = gr.Textbox()
    state = gr.State()

    message.submit(distribute, inputs=[model_selector, message, top_p, temperature, repetition_penalty, state],
                   outputs=[message, chatbot, state])

    with gr.Row():
        clear_history = gr.Button("🗑 Clear History")
        # clear = gr.Button('🧹 清除输入 | Clear Input')
        send = gr.Button("🚀 Send")
        regenerate = gr.Button("🔄 Regenerate")
        upvote_btn = gr.Button(value="👍  Upvote")
        downvote_btn = gr.Button(value="👎  Downvote")

    regenerate.click(regeneratefunc, state, [message, state], queue=False).then(
        distribute, inputs=[model_selector, message, top_p, temperature, repetition_penalty, state],
        outputs=[message, chatbot, state])

    send.click(distribute, inputs=[model_selector, message, top_p, temperature, repetition_penalty, state],
               outputs=[message, chatbot, state])
    # clear.click(lambda: None, None, message, queue=False)
    clear_history.click(clear_session, inputs=[], outputs=[message, chatbot, state], queue=False)
    upvote_btn.click(upvote_last_response, inputs=[state, model_selector], outputs=[message])
    downvote_btn.click(downvote_last_response, inputs=[state, model_selector], outputs=[message])

demo.queue(concurrency_count=20, max_size=None)
demo.launch(server_name="0.0.0.0", server_port=9997, debug=True, inbrowser=False, share=False)
# demo.launch(server_name="0.0.0.0", server_port=9997, debug=True, inbrowser=False, share=False,
#             auth=("siat-nlp", "siat-nlp"), auth_message="法律对话系统——韩非")
