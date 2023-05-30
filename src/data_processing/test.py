
# import re
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # line = '   你好 \n  你好   '
# # print(line)
# # line = re.sub(r'\s+', '', line)
# # print(line)


# def text2token(text_list):
#     tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/wjb/LawChatGPT/pretrain_model")

#     tokens = []
#     for text in text_list:
#         text = text + tokenizer.eos_token
#         token = tokenizer.encode(text)
#         tokens.extend(token)

#     print(len(tokens))
    
#     # print(tokens)
#     # print(tokenizer.decode(tokens))


# text_list = [
#     "审判长伍辉耘\n审判员徐伟\n审判员缪崇民\n二〇一三年三月二十七日\n书记员王佳",
#     "衢州市衢江区人口和计划生育局、衢州市衢江区人口和计划生育局依据已经发生法与邵红楼、汤明仙非诉执行审查裁定书",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。",
#     "上诉人深圳市规划和国土资源委员会因与被上诉人深圳市投资控股有限公司房地产权抵押登记纠纷一案，不服深圳市罗湖区人民法院（2011）深罗法行初字第8号行政判决，向本院提起上诉。本院依法组成合议庭进行了审理，本案现已审理终结。"
# ]
# text2token(text_list)

# content = re.sub(r'<[^>]+>', '', "aaa")

# # content = re.findall(r'<[^>]+>([^<]+)</[^>]+>', )

# print(content)


# res = []


# import os
# def gci(filepath):
#   files = os.listdir(filepath)
#   for fi in files:
#     fi_d = os.path.join(filepath, fi)            
#     if os.path.isdir(fi_d):
#       gci(fi_d)                  
#     else:
#       res.append(os.path.join(filepath, fi_d))

# gci('/mntcephfs/data/med/wjb/LawChatGPT/data/qa')

# print(res)
# print(len(res))


# list_1 = [
#   [1,2,3],
#   [5,43,62,4],
#     [2,2,3],
# ]

# import random
# random.shuffle(list_1)
# print(list_1)

print('\ndfdsafsa  \n dfa\n'.strip())