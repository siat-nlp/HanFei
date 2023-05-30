import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def is_empty(text):
    if text == "​" or text == '':
        return True
    return False

def filter_data(file_path):
    new_datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if filter(data):
                continue
            else:
                new_datas.append(data)

    return new_datas

    
def make_sample(sample, start_idx, end_idx, idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(idx),
        "conversations": sample["conversations"][start_idx:end_idx],
    }

def split_one_sample(sample, tokenizer, max_length, idx):
    tokenized_lens = []
    conversations = sample["conversations"]
    conversations = conversations[: len(conversations) // 2 * 2]

    if len(conversations) % 2 != 0 or len(conversations) < 2:
        return []

    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 6
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i, idx))
            start_idx = i
            cur_len = 0
            idx += 1
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2, idx))
            idx += 1
        cur_len += tmp_len

    return new_samples, idx


if __name__ == '__main__':
    
    total_finetune_data_cnt = 0
    # 法律指令
    datas = filter_data('./../data/raw_zh_law_instruction.json')
    new_datas = []
    for i, data in enumerate(datas):

        conversations = []

        if data['input'] == '':
            human = {
                'from': 'human',
                'value': data['instruction']
            }
        else:
            human = {
                'from': 'human',
                'value': data['instruction'] + '\n' + data['input']
            }

        gpt = {
            'from': 'gpt',
            'value': data['output']
        }

        conversations.append(human)
        conversations.append(gpt)

        id = "zh_law_instruction_{}".format(i+1)

        new_datas.append({'id': id, 'conversations': conversations})

    print('中文法律指令:{}'.format(len(new_datas)))

    total_finetune_data_cnt += len(new_datas)

    with open('./../data/zh_law_instruction.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)

    # 中文通用指令
    datas = []
    with open('./../data/Belle_open_source_1M.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 获取所有下标
        indices = list(range(len(lines)))
        # 打乱下标
        random.shuffle(indices)    
        
        for i in tqdm(indices[:30000]):
            data = json.loads(lines[i])
            if data['output'] == '':
                continue
            datas.append(data)

    with open('./../data/raw_zh_general_instruction.json', 'r', encoding='utf-8') as f:
        json_datas = json.load(f)
        for data in json_datas:
            if data['output'] == '':
                continue
            datas.append(data)

    new_datas = []

    for i, data in tqdm(enumerate(datas)):

        conversations = []

        if data['input'] == '':
            human = {
                'from': 'human',
                'value': data['instruction']
            }
        else:
            human = {
                'from': 'human',
                'value': data['instruction'] + '\n' + data['input']
            }

        gpt = {
            'from': 'gpt',
            'value': data['output']
        }

        conversations.append(human)

        conversations.append(gpt)

        id = "zh_general_instruction_{}".format(i+1)

        new_datas.append({'id': id, 'conversations': conversations})

    print('中文通用指令:{}'.format(len(new_datas)))

    total_finetune_data_cnt += len(new_datas)

    with open('./../data/zh_general_instruction.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)
    
    # 合同指令
    datas = []
    with open('./../data/raw_zh_contract_instruction.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            datas.append(data)

    new_datas = []

    for i, data in tqdm(enumerate(datas)):

        conversations = []

        if data['input'] == '':
            human = {
                'from': 'human',
                'value': data['instruction']
            }
        else:
            human = {
                'from': 'human',
                'value': data['instruction'] + '\n' + data['input']
            }

        gpt = {
            'from': 'gpt',
            'value': data['output']
        }

        conversations.append(human)

        conversations.append(gpt)

        id = "zh_contract_instruction_{}".format(i+1)

        new_datas.append({'id': id, 'conversations': conversations})

    print('中文合同指令:{}'.format(len(new_datas)))

    total_finetune_data_cnt += len(new_datas)

    with open('./../data/zh_contract_instruction.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)


    # 中文通用对话
    idx=1
    tokenizer = AutoTokenizer.from_pretrained("/data2/wenjiabao/HanFei/pretrain_model/bloomz-7b1-mt")
    new_datas = []
    with open('./../data/multiturn_chat_0.8M.json', "r") as file:
        lines = file.readlines()
        # 获取所有下标
        indices = list(range(len(lines)))
        # 打乱下标
        random.shuffle(indices)
        # 取前 50000 个
        indices = indices[:50000]
        for index in tqdm(indices):
            # 读取对应行
            line = lines[index]

            data = json.loads(line)

            instruction = data['instruction']

            output = data['output']

            instruction = instruction.replace('Assistant:', '||').replace('Human:', '||')

            conversations = instruction.split('||')

            if conversations[0] == '':
                conversations.pop(0)
            
            if conversations[-1] == '':
                conversations.pop(-1)

            conversations.append(output)

            conversations = [conversation.strip() for conversation in conversations]

            conversations = conversations[: len(conversations) // 2 * 2]

            bad_case = False

            new_conversations = []
            
            for i in range(0, len(conversations), 2):

                human_value = conversations[i]

                gpt_value = conversations[i + 1]

                if 'chatgpt' in human_value.lower() or 'chatgpt' in gpt_value.lower():
                    bad_case = True
                    break

                if is_empty(human_value) or is_empty(gpt_value):
                    continue

                human_role = 'human'
                gpt_role    = 'gpt'

                new_conversations.append({'from': human_role, 'value': human_value})
                new_conversations.append({'from': gpt_role, 'value': gpt_value})

            if bad_case:
                continue

            if len(new_conversations) > 0:
                news_samples, idx = split_one_sample({'id': 'zh_general_conversation', 'conversations': new_conversations}, tokenizer, 2048, idx)
                new_datas.extend(news_samples) 
        
    with open('./../data/raw_zh_en_general_conversation.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)
        zh_general_conversations = datas['g_conv_zh']

        for conversations in tqdm(zh_general_conversations):
            
            new_conversations = []
            
            conversations = conversations[: len(conversations) // 2 * 2]

            bad_case = False

            for i in range(0, len(conversations), 2):
                human = conversations[i]
                gpt = conversations[i + 1]

                human_role = human[0]
                human_value = human[1]

                gpt_role = gpt[0]
                gpt_value = gpt[1]

                if 'chatgpt' in human_value.lower() or 'chatgpt' in gpt_value.lower():
                    bad_case = True
                    break

                if human_role != '用户' or gpt_role != 'gpt':
                    bad_case = True
                    break

                if is_empty(human_value) or is_empty(gpt_value):
                    continue

                human_role = 'human'

                new_conversations.append({'from': human_role, 'value': human_value})
                new_conversations.append({'from': gpt_role, 'value': gpt_value})

            if bad_case:
                continue

            if len(new_conversations) > 0:
                news_samples, idx = split_one_sample({'id': 'zh_general_conversation', 'conversations': new_conversations}, tokenizer, 2048, idx)
                new_datas.extend(news_samples)
    
    print('中文通用对话:{}'.format(len(new_datas)))

    total_finetune_data_cnt += len(new_datas)

    with open('./../data/zh_general_conversation.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)

    # 中文法律对话
    tokenizer = AutoTokenizer.from_pretrained("/data2/wenjiabao/HanFei/pretrain_model/bloomz-7b1-mt")
    new_datas = []
    with open('./../data/raw_zh_law_conversation.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)
        
        idx = 1
        for conversations in tqdm(datas):
            
            new_conversations = []
            conversations = conversations['conversations']
            conversations = conversations[: len(conversations) // 2 * 2]

            bad_case = False

            for i in range(0, len(conversations), 2):
                human = conversations[i]
                gpt = conversations[i + 1]

                human_role = human['from']
                human_value = human['value']

                gpt_role = gpt['from']
                gpt_value = gpt['value']

                if 'chatgpt' in human_value.lower() or 'chatgpt' in gpt_value.lower():
                    bad_case = True
                    break

                if human_role != 'human' or gpt_role != 'gpt':
                    bad_case = True
                    break

                if is_empty(human_value):
                    continue

                if is_empty(gpt_value):
                    continue

                new_conversations.append({'from': human_role, 'value': human_value})

                new_conversations.append({'from': gpt_role, 'value': gpt_value})

            if bad_case:
                continue

            if len(new_conversations) > 0:
                news_samples, idx = split_one_sample({'id': 'zh_law_conversation', 'conversations': new_conversations}, tokenizer, 2048, idx)
                new_datas.extend(news_samples)
    
    print('中文法律对话:{}'.format(len(new_datas)))

    total_finetune_data_cnt += len(new_datas)

    with open('./../data/zh_law_conversation.json', 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)
    
    print('总数据量:{}'.format(total_finetune_data_cnt))


    # 混合所有数据
    train_datas = []
    val_datas = []
    val_cnt = 10000
    total_data_cnt = 0
    # with open('./../data/zh_contract_instruction.json', 'r', encoding='utf-8') as f:
    #     data1 = json.load(f)
    #     cnt1 = len(data1)
    #     total_data_cnt += cnt1
    with open('./../data/zh_general_conversation.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
        cnt2 = len(data2)
        total_data_cnt += cnt2

    with open('./../data/zh_general_instruction.json', 'r', encoding='utf-8') as f:
        data3 = json.load(f)
        cnt3 = len(data3)
        total_data_cnt += cnt3

    with open('./../data/zh_law_conversation.json', 'r', encoding='utf-8') as f:
        data4 = json.load(f)
        cnt4 = len(data4)
        total_data_cnt += cnt4

    with open('./../data/zh_law_instruction.json', 'r', encoding='utf-8') as f:
        data5 = json.load(f)
        cnt5 = len(data5)
        total_data_cnt += cnt5

    with open('./../data/zh_law_qa.json', 'r', encoding='utf-8') as f:
        data6 = json.load(f)    
        cnt6 = len(data6)
        total_data_cnt += cnt6

    # val_size_1 = int(cnt1 * val_cnt / total_data_cnt)
    val_size_2 = int(cnt2 * val_cnt / total_data_cnt)
    val_size_3 = int(cnt3 * val_cnt / total_data_cnt)
    val_size_4 = int(cnt4 * val_cnt / total_data_cnt)
    val_size_5 = int(cnt5 * val_cnt / total_data_cnt)
    val_size_6 = int(cnt6 * val_cnt / total_data_cnt)
    print(val_size_2, val_size_3, val_size_4, val_size_5, val_size_6)

    # index1s = list(range(len(data1)))
    # random.shuffle(index1s)
    index2s = list(range(len(data2)))
    random.shuffle(index2s)
    index3s = list(range(len(data3)))
    random.shuffle(index3s)
    index4s = list(range(len(data4)))
    random.shuffle(index4s)
    index5s = list(range(len(data5)))
    random.shuffle(index5s)
    index6s = list(range(len(data6)))
    random.shuffle(index6s)

    # for idx in index1s[:val_size_1]:
    #     val_datas.append(data1[idx])
    for idx in index2s[:val_size_2]:
        val_datas.append(data2[idx])
    for idx in index3s[:val_size_3]:
        val_datas.append(data3[idx])
    for idx in index4s[:val_size_4]:
        val_datas.append(data4[idx])
    for idx in index5s[:val_size_5]:
        val_datas.append(data5[idx])
    for idx in index6s[:val_size_6]:
        val_datas.append(data6[idx])
    
    # for idx in index1s[val_size_1:]:
    #     train_datas.append(data1[idx])
    for idx in index2s[val_size_2:]:
        train_datas.append(data2[idx])
    for idx in index3s[val_size_3:]:
        train_datas.append(data3[idx])
    for idx in index4s[val_size_4:]:
        train_datas.append(data4[idx])
    for idx in index5s[val_size_5:]:
        train_datas.append(data5[idx])
    for idx in index6s[val_size_6:]:
        train_datas.append(data6[idx])

    print('shuffle...')
    indexs = list(range(len(train_datas)))
    random.shuffle(indexs)
    new_train_datas = []
    for idx in indexs:
        new_train_datas.append(train_datas[idx])
    train_datas = new_train_datas

    indexs = list(range(len(val_datas)))
    random.shuffle(indexs)
    new_val_datas = []
    for idx in indexs:
        new_val_datas.append(val_datas[idx])
    val_datas = new_val_datas
    

    print('总数据量:{}'.format(len(train_datas) + len(val_datas)))
    print('训练数据量:{}'.format(len(train_datas)))
    print('验证数据量:{}'.format(len(val_datas)))

    with open('./../data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_datas, f, ensure_ascii=False, indent=4)
    with open('./../data/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_datas, f, ensure_ascii=False, indent=4)

    # 过滤太长的对话
    new_json_datas = []
    with open('./../data/train.json', 'r', encoding='utf-8') as f:
        json_datas = json.load(f)
        print(len(json_datas))
        for ins in tqdm(json_datas):
            conversations = ins['conversations']
            if len(conversations) == 0:
                continue
            try:
                assert len(conversations) % 2 == 0
            except:
                print(ins['id'])
            not_to_long = True
            for i in range(0, len(conversations), 2):
                try:
                    assert conversations[i]['from'] == 'human'
                    assert conversations[i]['value'] != ''
                    assert conversations[i+1]['from'] == 'gpt'
                    assert conversations[i+1]['value'] != ''
                    human_value = conversations[i]['value']
                    gpt_value = conversations[i+1]['value']
                    human_token_length = len(tokenizer(human_value).input_ids)
                    gpt_token_length = len(tokenizer(gpt_value).input_ids)

                    if human_token_length + gpt_token_length > 2000:
                        not_to_long = False
                        break
                except:
                    print(ins['id'])
            if not_to_long:
                new_json_datas.append(ins)
    print('训练数据筛选结果：', len(new_json_datas))
    with open('./../data/train.json', 'w', encoding='utf-8') as f:
        json.dump(new_json_datas, f, ensure_ascii=False, indent=4)

    new_json_datas = []
    with open('./../data/val.json', 'r', encoding='utf-8') as f:
        json_datas = json.load(f)
        print(len(json_datas))
        for ins in tqdm(json_datas):
            conversations = ins['conversations']
            if len(conversations) == 0:
                continue
            try:
                assert len(conversations) % 2 == 0
            except:
                print(ins['id'])
            not_to_long = True
            for i in range(0, len(conversations), 2):
                try:
                    assert conversations[i]['from'] == 'human'
                    assert conversations[i]['value'] != ''
                    assert conversations[i+1]['from'] == 'gpt'
                    assert conversations[i+1]['value'] != ''
                    human_value = conversations[i]['value']
                    gpt_value = conversations[i+1]['value']
                    human_token_length = len(tokenizer(human_value).input_ids)
                    gpt_token_length = len(tokenizer(gpt_value).input_ids)
                    if human_token_length + gpt_token_length > 2000:
                        not_to_long = False
                        break
                except:
                    print(ins['id'])
            if not_to_long:
                new_json_datas.append(ins)
    print('验证数据筛选结果：', len(new_json_datas))
    with open('./../data/val.json', 'w', encoding='utf-8') as f:
        json.dump(new_json_datas, f, ensure_ascii=False, indent=4)