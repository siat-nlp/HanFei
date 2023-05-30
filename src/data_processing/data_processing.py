import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import Pool
import random
import rich.progress


tokenizer = AutoTokenizer.from_pretrained("/mntcephfs/data/med/wjb/LawChatGPT/pretrain_model")

main_key_list = ['title', 'litigant', 'primaryFact', 'caseIntroduction', 'viewPoint', 'judgeResult'] # 主要字段

key_list = ['title', 'litigant', 'primaryFact', 'caseIntroduction', 'viewPoint', 'judgeResult', 'appendix', 'judgeJury'] # 所有字段

law_key_list = ['activeDate', 'content', 'issuedNo', 'lawType', 'levelType', 'lockFlag', 'publishDate', 'publisherName', 'relatedLawName', 'timelinessType', 'title', 'regionName']

new_key_list = ['title', 'content', 'region', 'industry']

qa_key_list = ['query', 'answer', 'domain', 'category'] 

viewpoint_key_list = ['title', 'author', 'publishTime', 'periodicalYear', 'wxPublicName', 'summary', 'content', 'lockFlag'] 

prosecution_document_key_list = ['title', 'primaryFact', 'viewPoint', 'badging', 'appendix', 'proDate', 'docNo', 'docType', 'proName', 'causeName']

administrative_punishment_key_list = ['title', 'content', 'pDate', 'pNo', 'regionName', 'pOrganization']

random.seed(42)

def splitDoc(doc):
    return doc.split('\002')

def split2attrs(case):
    return case.split('\001')

def deleteBlank(text):
    return re.sub(r'\s+', '', text)

def cal_doc_avg_len():
    return 0

def is_verdict(data):
    '''
    判断是否为判决书
    '''
    if data['documentTypeName'] == '判决书':
        return True
    else:
        return False

def is_importance_court(data):
    '''
    判断是否为重要法院
    '''
    # 取字符串最后6个字符
    courtName = data['courtName'][-6:]
    importance_court_list = ['中级人民法院', '高级人民法院', '最高人民法院']
    if courtName in importance_court_list:
        return True
    else:
        return False

def has_main_field(data):
    for key in main_key_list:
        if data[key] == '':
            return False
    return True

def raw_data2json(fold_name, input_fold, output_fold, start_year, end_year):

    for year in range(start_year, end_year):
        # 连接路径
        input_dir = os.path.join(input_fold, str(year))
        output_dir = os.path.join(output_fold, str(year))

        # 如果目录不存在则跳过
        if not os.path.exists(input_dir):
            continue

        # 读取目录下所有文件
        filenames = os.listdir(input_dir)

        case_set = set()

        # 变量记录文件数
        file_count = 0
        # 变量记录重复数据
        repeat_count = 0
        # 变量记录不重复数据
        unique_count = 0

        for filename in tqdm(filenames, desc='raw_data2json {} {} year'.format(fold_name, year)):
            # 从路径中获取文件名
            input_file = os.path.join(input_dir, filename)
            # 将文件改为json后缀
            prefix = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, prefix+'.json')
            # 如果没有该文件夹则创建
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            case = []
            with open(input_file,'r')as f:
                # 读取整个文件
                lines = f.read()
                lines = splitDoc(lines)

                for line in lines:
                    # 正则删除文本中所有的空格
                    # line = deleteBlank(line)

                    attrs = split2attrs(line)

                    doc = ''.join(attrs)
                    file_count += 1
                    if doc in case_set:
                        repeat_count += 1
                        continue
                    else:
                        unique_count += 1

                        case_set.add(doc)

                        appendix = attrs[0]
                        # 案情描述
                        caseIntroduction = attrs[1]
                        # 案号
                        caseNo = attrs[2]
                        # 案件类型
                        caseTypeName = attrs[3]
                        # 案由名称
                        causeName = attrs[4]
                        # 法院名称
                        courtName = attrs[5]
                        # 被告律师
                        defendantLawyer = attrs[6]
                        # 文书类型
                        documentTypeName = attrs[7]
                        # 裁判日期
                        judgeDate = attrs[8]
                        # 审判人员
                        judgeJury = attrs[9]
                        # 法官数组
                        judgeList = attrs[10]
                        # 裁判结果
                        judgeResult = attrs[11]
                        # 律所数组
                        lawyerFirmList = attrs[12]
                        # 律师数组
                        lawyerList = attrs[13]
                        # 当事人信息
                        litigant = attrs[14]
                        # 第三人律师
                        otherLawyer = attrs[15]
                        # 原告律师
                        plaintiffLawyer = attrs[16]
                        # 基本事实
                        primaryFact = attrs[17]
                        # 发布日期
                        publishDate = attrs[18]
                        # 案例标题
                        title = attrs[19]
                        # 审理程序
                        trialTypeName = attrs[20]
                        # 法院认为
                        viewPoint = attrs[21]
                        # 发布类型
                        publishType = attrs[22]

                        # 将所有数据转化为json格式
                        caseContent = {
                            "appendix": appendix,
                            "caseIntroduction": caseIntroduction,
                            "caseNo": caseNo,
                            "caseTypeName": caseTypeName,
                            "causeName": causeName,
                            "courtName": courtName,
                            "defendantLawyer": defendantLawyer,
                            "documentTypeName": documentTypeName,
                            "judgeDate": judgeDate,
                            "judgeJury": judgeJury,
                            "judgeList": judgeList,
                            "judgeResult": judgeResult,
                            "lawyerFirmList": lawyerFirmList,
                            "lawyerList": lawyerList,
                            "litigant": litigant,
                            "otherLawyer": otherLawyer,
                            "plaintiffLawyer": plaintiffLawyer,
                            "primaryFact": primaryFact,
                            "publishDate": publishDate,
                            "title": title,
                            "trialTypeName": trialTypeName,
                            "viewPoint": viewPoint,
                            'publishType': publishType
                        }
                        case.append(caseContent)

            # 所有数据写入json文件
            with open(output_file, 'w') as f:
                json.dump(case, f, ensure_ascii=False, indent=4)

        print(unique_count)

        print(repeat_count)

        print(file_count)

def select_case(fold_name, input_fold, output_fold, start_year, end_year):
    # 要求:
    # 1.主要字段非空
    # 2.指导性案例、公报案例、典型案例。这个条件没用，基本上都是普通案例
    # 3.2017-2023年的普通案例
    # 4.文书类型为判决书
    # 5.法院层级为中级人民法院、高级人民法院、最高人民法院
    case_count = 0
    for year in range(start_year, end_year):
        # 连接路径
        input_dir = os.path.join(input_fold, str(year))
        # 如果目录不存在则跳过
        if not os.path.exists(input_dir):
            continue
        # 读取目录下所有文件
        filenames = os.listdir(input_dir)
        cases = []
        for filename in tqdm(filenames, desc='{} select {} year train case'.format(fold_name, year)):
            # 从路径中获取文件名
            input_file = os.path.join(input_dir, filename)
            with open(input_file,'r')as f:
                # 读取整个文件
                json_datas = json.load(f)
                for data in json_datas:
                    # 不是判决书、不是重要法院、主要字段为空，舍弃
                    if is_verdict(data) == False or is_importance_court(data) == False or has_main_field(data) == False:
                        continue 
                    cases.append(data)  
        case_count += len(cases)
        # 保存数据
        save_data(cases, output_fold, year)
    print('case_count: ', case_count)

def law2text(data):
    text = data['title'] + '\n'

    if data['publisherName']!= '':
        text = text + '发文机关：' + data['publisherName'] + '\n'
    
    if data['issuedNo']!= '':
        text = text + '发文字号：' + data['issuedNo'] + '\n'

    if data['levelType']!= '':
        text = text + '效力级别：' + data['levelType'] + '\n'

    if data['publishDate']!= '':
        text = text + '发布日期：' + data['publishDate'] + '\n'
    
    if data['activeDate']!= '':
        text = text + '实施日期：' + data['activeDate'] + '\n'

    if data['timelinessType']!= '':
        text = text + '时效性：' + data['timelinessType'] + '\n'

    if data['lawType']!= '':
        text = text + '法规类型：' + data['lawType'] + '\n'
    
    if data['regionName']!= '':
        text = text + '地域：' + data['regionName'] + '\n'
    
    if data['content']!= '':
        text = text + data['content']

    return text

def qa2text(data):
    text = data['query'] + '\n' + data['answer']
    text = text.replace(' ', '')
    text = text.replace('\n\n', '\n')
    return text

def prosecution_document2text(data):

    text = data['title'] + '\n'

    if data['docNo'] != '':
        text = text + '文书号：' + data['docNo'] + '\n'
    
    if data['docType'] != '':
        text = text + '文书类型：' + data['docType'] + '\n'
    
    if data['proName'] != '':
        text = text + '检察院：' + data['proName'] + '\n'
    
    if data['causeName'] != '':
        text = text + '案由：' + data['causeName'] + '\n'

    if data['proDate'] != '':
        text = text + '发布日期：' + data['proDate'] + '\n'

    if data['primaryFact'] != '':
        text = text + data['primaryFact']
    
    if data['viewPoint'] != '':
        text = text + data['viewPoint']
    
    if data['appendix'] != '':
        text = text + data['appendix'] 
    
    if data['badging'] != '':
        text = text + data['badging']
    
    return text


def viewpoint2text(data):
    text = data['title'] + '\n'

    # if data['author'] != '':
    #     text = text + '作者：' + data['author'] + '\n'
    
    if data['publishTime'] != '':
        text = text + '发布时间：' + data['publishTime'] + '\n'
    
    if data['summary'] != '':
        text = text + '摘要：' + data['summary'] + '\n'
    
    if data['content'] != '':
        text = text + data['content']

    return text

def new2text(data):
    text = data['title'] + '\n'

    if data['region'] != '':
        text = text + '地域：' + data['region'] + '\n'
    
    if data['industry'] != '':
        text = text + '行业：' + data['industry'] + '\n'
    
    if data['content'] != '':
        text = text + data['content']
    
    return text

def administrative_punishment2text(data):

    text = data['title'] + '\n'

    if data['pDate'] != '':
        text = text + '时间：' + data['pDate'] + '\n'
    
    if data['pNo'] != '':
        text = text + '处罚号：' + data['pNo'] + '\n'

    if data['regionName'] != '':
        text = text + '地域：' + data['regionName'] + '\n' 
    
    if data['pOrganization'] != '':
        text = text + '处罚机构：' + data['pOrganization'] + '\n' 

    if data['content'] != '':
        text = text + data['content']

    return text

def case2text(case):
    # 将案例转化为一行
    text = case['title'] + '\n'

    courtName = case['courtName']
    if courtName != '':
        text = text + '审理法院：' + courtName + '\n'

    caseTypeName = case['caseTypeName']
    if caseTypeName != '':
        text = text + '案件类型：' + caseTypeName + '\n'

    causeName = case['causeName']
    if causeName != '':
        text = text + '案由：' + causeName + '\n'
    
    caseNo = case['caseNo']
    if caseNo != '':
        text = text + '案号：' + caseNo + '\n'

    trialTypeName = case['trialTypeName']
    if trialTypeName != '':
        text = text + '审理程序：' + trialTypeName + '\n'

    judgeDate = case['judgeDate']
    if judgeDate != '':
        text = text + '裁判日期：' + judgeDate + '\n'

    publishDate = case['publishDate']
    if publishDate != '':
        text = text + '发布日期：' + publishDate + '\n'

    text = text + case['litigant'] + case['primaryFact'] + case['caseIntroduction'] + case['viewPoint'] + case['judgeResult'] + case['appendix'] + case['judgeJury']

    return text

def process(i, datas, toTextFun):
    res = []
    for data in tqdm(datas, desc='process {} start'.format(i)):
        text = toTextFun(data) + tokenizer.eos_token
        token = tokenizer.encode(text)
        data = {'text': text, 'token': token}
        res.append(data)
    return res

# 多进程处理
def multi_process(datas, process_num, toTextFun):

    # 每个进程处理的数据量
    process_size = len(datas) // process_num
        
    pool = Pool(process_num)
    multi_result = []
    # 开始运行
    for i in range(process_num):
        # 最后一个进程处理剩余的数据
        if i == process_num - 1:
            process_cases = datas[i * process_size:]
            multi_result.append(pool.apply_async(func=process, args=(i, process_cases, toTextFun)))
        else:
            process_cases = datas[i * process_size: (i + 1) * process_size]
            multi_result.append(pool.apply_async(func=process, args=(i, process_cases, toTextFun)))

    pool.close()
    pool.join()
    
    datas = []
    for res in multi_result:
        datas.extend(res.get())

    return datas

def case2TextAndToken(fold_name, input_fold, output_fold, start_year, end_year):
    # 将案例转化为文本
    case_count = 0
    for year in range(start_year, end_year):
        # 连接路径
        input_dir = os.path.join(input_fold, str(year))
        # 如果目录不存在则跳过
        if not os.path.exists(input_dir):
            continue
        # 读取目录下所有文件
        filenames = os.listdir(input_dir)
        json_datas = []
        for filename in tqdm(filenames, desc='{} train case to text and token {} year'.format(fold_name, year)):
            # 从路径中获取文件名
            input_file = os.path.join(input_dir, filename)
            with open(input_file, 'r')as f:
                # 读取整个文件
                json_datas.extend(json.load(f))
        
        datas = multi_process(json_datas, 16, case2text)

        case_count += len(datas)
        # 保存数据
        # save_data(cases, output_fold, year, indent=0)

        for i in range(0, len(datas), 1000):
            output_file = os.path.join(output_fold, str(year), str(i) + '.json')
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            with open(output_file, 'w', encoding='utf-8') as fp:
                fp.write('[' +',\n'.join(json.dumps(item, ensure_ascii=False) for item in datas[i:i+1000]) +']\n')

    print('case_count: ', case_count)

def save_data(cases, output_fold, year, indent=4):
    # 保存数据
    # 每1000个案例保存一次
    for i in range(0, len(cases), 1000):
        output_file = os.path.join(output_fold, str(year), str(i) + '.json')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w') as f:
            json.dump(cases[i:i+1000], f, ensure_ascii=False, indent=indent)

def getTraintoken(input_file_list, output_fold):
    tokens_list = []
    token_queue = []
    max_len = 2048
    batch_size = 100000 # 处理一次
    # 文件大小达到一定程度，停止
    cnt=0
    for i, input_file in enumerate(tqdm(input_file_list, desc='to token')):
        with open(input_file, 'r')as f:
            # 读取整个文件
            json_datas=json.load(f)            
            for data in json_datas:
                # 判断中文比例
                zh_cnt = cnt_zh(data['text'])
                # 如果汉字比例小于0.5则跳过
                if zh_cnt / len(data['text']) < 0.5:
                    continue

                tokens_list.append(data['token'])

        if len(tokens_list) >= batch_size or i == len(input_file_list) - 1: 
            
            idxs = list(range(len(tokens_list)))

            random.shuffle(idxs)
            output_file = os.path.join(output_fold, str(cnt)+'.json')
            with open(os.path.join(output_file), 'a')as fp:
                for idx in idxs:
                    token_queue.extend(tokens_list[idx])
                    while len(token_queue) >= max_len:
                        fp.write(json.dumps({'token':token_queue[:max_len]}, ensure_ascii=False)+'\n')
                        token_queue = token_queue[max_len:]
            # 输出文件大小
            print('file size {}'.format(os.path.getsize(output_file)/1024/1024))

            tokens_list = []
            cnt += 1
         

def data2json(input_fold, output_fold, key_list):
    # 读取目录下所有文件
    filenames = os.listdir(input_fold)
    for filename in tqdm(filenames, desc='data to json'):
        # 从路径中获取文件名
        input_file = os.path.join(input_fold, filename)
        prefix = filename.split('.')[0]
        output_file = os.path.join(output_fold, prefix+'.json')
        with open(input_file, 'r')as f:
            # 读取整个文件
            docs = f.read()
            docs = splitDoc(docs)
            doc_list = []
            for doc in docs:
                doc = split2attrs(doc)
                
                data = {}

                for i in range(len(doc)):
                    data[key_list[i]] = doc[i]
                
                doc_list.append(data)
            # 所有数据写入json文件
            with open(output_file, 'w') as f:
                json.dump(doc_list, f, ensure_ascii=False, indent=4)
def is_zh(c):
    if '\u4e00' <= c <= '\u9fff':
        return True
    return False

def cnt_zh(text):
    cnt = 0
    for c in text:
        if is_zh(c):
            cnt += 1
    return cnt

def clean_html_label(text):
    return re.sub(r'<[^>]+>', '', text)



def get_high_quality_data(input_fold, output_fold, title_name, main_key_list, min_len, data_type=None):

    def save_file(cnt, output_fold, tmps):
        output_file = os.path.join(output_fold, str(cnt) + '.json')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w') as sf:
            json.dump(tmps, sf, ensure_ascii=False, indent=4)

    filenames = os.listdir(input_fold)
    datas = []
    batch_size = 1000
    cnt=0
    for filename in tqdm(filenames, desc='select high quality data'):
        # 从路径中获取文件名
        input_file = os.path.join(input_fold, filename)
        with open(input_file,'r')as f:
            # 读取整个文件
            json_datas = json.load(f)
            for data in json_datas:
                if data[title_name] == '': # 去掉标题是空的
                    continue

                if data_type == 'laws':  # 法规过滤规则
                    if data['timelinessType'] == '失效':
                        continue
                flag = False

                for main_key in main_key_list:
                    # 删除网页标签
                    data[main_key] = clean_html_label(data[main_key])
                    if len(data[main_key]) < min_len:
                        flag = True
                        break
                    # 统计汉字的个数
                    zh_cnt = cnt_zh(data[main_key])
                    # 如果汉字比例小于0.5则跳过
                    if zh_cnt / len(data[main_key]) < 0.5:
                        flag = True
                        break
                if flag:
                    continue
                datas.append(data)  
        # 每1000个案例保存一次
        while len(datas) > batch_size:
            cnt += 1
            save_file(cnt, output_fold, datas[:batch_size])
            datas = datas[batch_size:]
    # 每1000个案例保存一次
    while len(datas) > batch_size:
        cnt += 1
        save_file(cnt, output_fold, datas[:batch_size])
        datas = datas[batch_size:]

# def select_train_data(input_fold, output_fold, title_name, main_key_list, min_len):
#     filenames = os.listdir(input_fold)
#     datas = []
#     for filename in tqdm(filenames, desc='select data'):
#         # 从路径中获取文件名
#         input_file = os.path.join(input_fold, filename)
    #     with open(input_file,'r')as f:
    #         # 读取整个文件
    #         json_datas = json.load(f)
    #         for data in json_datas:
    #             if data[title_name] == '':
    #                 continue
                
    #             flag = False

    #             for main_key in main_key_list:
    #                 # 删除网页标签
    #                 data[main_key] = clean_html_label(data[main_key])
    #                 if len(data[main_key]) < min_len:
    #                     flag = True
    #                     break
    #                 # 统计汉字的个数
    #                 zh_cnt = cnt_zh(data[main_key])
    #                 # 如果汉字比例小于0.5则跳过
    #                 if zh_cnt / len(data[main_key]) < 0.5:
    #                     flag = True
    #                     break
    #             if flag:
    #                 continue

    #             datas.append(data)  
    # # 每1000个案例保存一次
    # for i in range(0, len(datas), 1000):
    #     output_file = os.path.join(output_fold, str(i) + '.json')
    #     if not os.path.exists(os.path.dirname(output_file)):
    #         os.makedirs(os.path.dirname(output_file))
    #     with open(output_file, 'w') as f:
    #         json.dump(datas[i:i+1000], f, ensure_ascii=False, indent=4)

def ToTextAndToken(input_fold, output_fold, toTextFun):
    # 读取目录下所有文件
    filenames = os.listdir(input_fold)
    json_datas = []
    for filename in tqdm(filenames, desc='data to text and token'):
        # 从路径中获取文件名
        input_file = os.path.join(input_fold, filename)
        with open(input_file, 'r')as f:
            # 读取整个文件
            json_datas.extend(json.load(f))
    
    datas = multi_process(json_datas, 32, toTextFun)

    for i in tqdm(range(0, len(datas), 1000)):
        output_file = os.path.join(output_fold, str(i) + '.json')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        with open(output_file, 'w', encoding='utf-8') as fp:
            fp.write('[' +',\n'.join(json.dumps(item, ensure_ascii=False) for item in datas[i:i+1000]) +']\n')



def select_train_data(raw_dir, data_type_list):

    paths = {}
    def gci(filepath, data_type):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath, fi)
            if os.path.isdir(fi_d):
                gci(fi_d, data_type)                  
            else:
                paths[data_type].append(os.path.join(filepath, fi_d))

    # 设置每个类别大小
    data_type_size = {}
    data_type_size['laws'] = 1024 * 1024 * 1024 * 60
    data_type_size['case'] = 1024 * 1024 * 1024 * 9
    data_type_size['qa'] = 1024 * 1024 * 1024 * 9
    data_type_size['news'] = 1024 * 1024 * 1024 * 9
    data_type_size['prosecution_document'] = 1024 * 1024 * 1024 * 9
    data_type_size['viewpoint'] = 1024 * 1024 * 1024 * 9
    data_type_size['administrative_punishment'] = 1024 * 1024 * 1024 * 9

    for data_type in data_type_list:
        # 加载每个类别的所有数据
        paths[data_type] = []
        input_fold = os.path.join(raw_dir, data_type, 'textAndtoken')
        # 获取所有文件
        gci(input_fold, data_type)

        random.shuffle(paths[data_type])

        textAndToken_list = []
        tmp_size = 0
        for file in tqdm(paths[data_type], desc='loading {}'.format(data_type)):
            tmp_size += os.path.getsize(file)
            with open(file, 'r') as f:
                data = json.load(f)
                textAndToken_list.extend(data)
            if tmp_size > data_type_size[data_type]:
                break

        # shuffle
        # idxs = list(range(len(textAndToken_list)))
        # random.shuffle(idxs)

        # 随机取到达到大小要求为止
        # text_list = []
        # token_list = []
        # for idx in tqdm(idxs, desc='select train data'):
        #     text_list.append(textAndToken_list[idx]['text'])
        #     token_list.append(textAndToken_list[idx]['token'])
        
        # 保存文件
        # assert len(text_list) == len(token_list)
        output_file = os.path.join(raw_dir, data_type, 'selected_data_1.json')

        with open(output_file, 'w') as f:
            for textAndToken in tqdm(textAndToken_list, desc='{} save select token'.format(data_type)):

                f.write(json.dumps(textAndToken, ensure_ascii=False) + '\n')
            
    # for data_type in data_type_list:
    #     print(data_type, len(paths[data_type]))
    
    # # 获取总的文件大小
    # 

    # for data_type in data_type_list:
    #     data_type_size[data_type] = 0
    #     for path in paths[data_type]:
    #         data_type_size[data_type] += os.path.getsize(path)
    
    # for key in data_type_list:
    #     print(key, data_type_size[key]/1024/1024/1024, 'GB')
    
    # # 计算比例
    # total_size = 0
    # for key in data_type_list:
    #     total_size += data_type_size[key]
    
    # for key in data_type_list: # 计算比例
    #     data_type_size[key] = data_type_size[key] / total_size
    #     print(key, data_type_size[key])

    # total_size = 60 * 1024 * 1024 * 1024  # B

    # # 分配比例
    # for key in data_type_list:
    #     data_type_size[key] = int(total_size * (1.0/7)) # B
    


    # for key in data_type_list:
    #     print(key, data_type_size[key]/1024/1024/1024, 'GB')

    # # 随机选择, 按比例划分
    # for data_type in data_type_list:
    #     random.shuffle(paths[data_type])
    #     tmp_list = []
    #     tmp_size = 0
    #     for path in paths[data_type]:
    #         tmp_size += os.path.getsize(path)
    #         tmp_list.append(path)
    #         if tmp_size >= data_type_size[data_type]:
    #             break
    #     paths[data_type] = tmp_list[:]

    # return paths    

def genTrainData(fold, train_file, valid_file, valid_size):
    # 列出所有文件
    filenames = os.listdir(fold)
    # 从文件中均匀取数据放到验证集中
    pre_file_valid_size = valid_size // len(filenames)

    first_file_valid_size = pre_file_valid_size + valid_size % len(filenames)

    with open(train_file, 'w') as tf:
        with open(valid_file, 'w') as vf:
            for i, filename in enumerate(tqdm(filenames, desc='gen train data')):
                input_file = os.path.join(fold, filename)
                with open(input_file, 'r') as f:
                    # 读取整个文件
                    datas = f.readlines()
                    if i == 0:
                        for data in datas[:first_file_valid_size]:
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            vf.write(data)
                        for data in datas[:]: # 训练集和验证集要重合
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            tf.write(data)
                    else:
                        for data in datas[:pre_file_valid_size]:
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            vf.write(data)
                        for data in datas[:]: # 训练集和验证集要重合
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            tf.write(data)

def join_select_data(raw_fold, data_type_list, train_file_path, valid_file_path, valid_size):
    lines = []
    for data_type in data_type_list[:]:
        output_file = os.path.join(raw_fold, data_type, 'selected_data_1.json')
        with rich.progress.open(file=output_file, mode="r", description='load {}'.format(data_type)) as f:
            lines.extend(f.readlines())
    
    idxs = list(range(len(lines)))
    random.shuffle(idxs)
    max_len = 2048

    # val
    tokens = []
    val_cnt = 0    
    with open(valid_file_path, 'w') as f:
        for idx in idxs:
            data = json.loads(lines[idx])
            tokens.extend(data['token'])
            while len(tokens) >= max_len and val_cnt < valid_size:
                f.write(json.dumps({"token":tokens[:max_len]}, ensure_ascii=False)+'\n')
                tokens = tokens[max_len:]
                val_cnt += 1
            
            if val_cnt >= valid_size:
                break

    # train
    tokens = []
    batch_size = 1000
    with open(train_file_path, 'w') as f:
        for idx in tqdm(idxs, desc='join_select_data'):
            data = json.loads(lines[idx])
            tokens.extend(data['token'])
            while len(tokens) >= max_len * batch_size:
                f.write('\n'.join(json.dumps({"token":tokens[i*max_len:(i+1)*max_len]}) for i in range(batch_size)) + '\n')
                tokens = tokens[max_len * batch_size:]
        while len(tokens) >= max_len:
            f.write(json.dumps({"token":tokens[:max_len]})+'\n')
            tokens = tokens[max_len:]


if __name__ == '__main__': 

    # 处理案例
    raw_case_dir = '/mntcephfs/data/med/wjb/data/case'   
    case_output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/case'
    
    case_fold_list = [
        'administrative_case2', 
        'civil_case2',
        'criminal_cases2',
        'execution_of_cases2',
        'national_compensation_cases2'
    ]

    # for fold in case_fold_list:
    #     raw_data_fold = os.path.join(raw_case_dir, fold)
    #     json_fold = os.path.join(case_output_fold, fold+'_json') # 转json
    #     selected_fold = os.path.join(case_output_fold, fold+'_select') # 选取
    #     textAndtoken_fold = os.path.join(case_output_fold, 'textAndtoken', fold) # textAndtoken
    #     token_fold = os.path.join(case_output_fold, 'token', fold) # token
        
    #     getTraintoken(fold, textAndtoken_fold, token_fold, 2022, 2024)

        # raw_data2json(fold, raw_data_fold, json_fold, 2014, 2024)
        # select_case(fold, json_fold, selected_fold, 2014, 2024)
        # case2TextAndToken(fold, selected_fold, textAndtoken_fold, 2020, 2024)
        
    
    # 处理法规
    # raw_dir = '/mntcephfs/data/med/wjb/data/laws'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/laws'
    # json_fold = os.path.join(output_fold, 'json')
    # highQuality_fold = os.path.join(output_fold, 'highQuality')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(highQuality_fold):
    #     os.makedirs(highQuality_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)
    # # data2json(raw_dir, json_fold, law_key_list)
    # # get_high_quality_data(json_fold, highQuality_fold, 'title', ['content'], 100, 'laws')
    # ToTextAndToken(highQuality_fold, textAndtoken_fold, law2text)

    # 处理新闻
    # raw_dir = '/mntcephfs/data/med/wjb/data/news'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/news'
    # json_fold = os.path.join(output_fold, 'json')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)    
    # data2json(raw_dir, json_fold, new_key_list)
    # select_data(json_fold, select_fold, 'title', ['content'], 100)
    # ToTextAndToken(select_fold, textAndtoken_fold, new2text)

    # 处理prosecution_document
    # raw_dir = '/mntcephfs/data/med/wjb/data/prosecution_document'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/prosecution_document'
    # json_fold = os.path.join(output_fold, 'json')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)    
    # data2json(raw_dir, json_fold, prosecution_document_key_list)
    # select_data(json_fold, select_fold, 'title', ['primaryFact', 'viewPoint'], 50)
    # ToTextAndToken(select_fold, textAndtoken_fold, prosecution_document2text)

    # 处理qa
    # raw_dir = '/mntcephfs/data/med/wjb/data/qa'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/qa'
    # json_fold = os.path.join(output_fold, 'json')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)    
    # data2json(raw_dir, json_fold, qa_key_list)
    # select_data(json_fold, select_fold, 'query', ['answer'], 10)
    # ToTextAndToken(select_fold, textAndtoken_fold, qa2text)

    # 处理viewpoint
    # raw_dir = '/mntcephfs/data/med/wjb/data/viewpoint'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/viewpoint'
    # json_fold = os.path.join(output_fold, 'json')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)    
    # data2json(raw_dir, json_fold, viewpoint_key_list)
    # select_data(json_fold, select_fold, 'title', ['content'], 150)
    # ToTextAndToken(select_fold, textAndtoken_fold, viewpoint2text)
    
    # # 处理administrative_punishment
    # raw_dir = '/mntcephfs/data/med/wjb/data/administrative_punishment'
    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/administrative_punishment'
    # json_fold = os.path.join(output_fold, 'json')
    # select_fold = os.path.join(output_fold, 'select')
    # textAndtoken_fold = os.path.join(output_fold, 'textAndtoken') # textAndtoken
    # token_fold = os.path.join(output_fold, 'token') # token
    # # 如果没有目录，则创建
    # if not os.path.exists(output_fold):
    #     os.makedirs(output_fold)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    # if not os.path.exists(select_fold):
    #     os.makedirs(select_fold)
    # if not os.path.exists(textAndtoken_fold):
    #     os.makedirs(textAndtoken_fold)
    # if not os.path.exists(token_fold):
    #     os.makedirs(token_fold)    
    # data2json(raw_dir, json_fold, administrative_punishment_key_list)
    # select_data(json_fold, select_fold, 'title', ['content'], 50)
    # ToTextAndToken(select_fold, textAndtoken_fold, administrative_punishment2text)

    # textAndtoken_fold_list = [
    #     '/mntcephfs/data/med/wjb/LawChatGPT/data/viewpoint/textAndtoken',
    # ]

    raw_dir = '/mntcephfs/data/med/wjb/LawChatGPT/data'

    data_type_list = ['laws', 'qa', 'viewpoint', 'administrative_punishment', 'case', 'prosecution_document', 'news']

    pool = Pool(len(data_type_list))
    # 开始运行
    for i in range(len(data_type_list)):
        pool.apply_async(func=select_train_data, args=(raw_dir, data_type_list[i:i+1]))
    pool.close()
    pool.join()

    # 将选中的数据合并起来
    join_select_data(raw_dir, data_type_list, './train_data_v3/train.json', './train_data_v3/val.json', 3840)

    # all_textAndToken_file_list = []
    # for key in selected_textAndToken_file_dict:
    #     all_textAndToken_file_list += selected_textAndToken_file_dict[key]

    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/train_token_v2'

    # # shuffle所有文件
    # random.shuffle(all_textAndToken_file_list)

    # getTraintoken(all_textAndToken_file_list, output_fold)

    # output_fold = '/mntcephfs/data/med/wjb/LawChatGPT/data/train_token_v2'

    # genTrainData(output_fold, './train_data_v2/train.json', './train_data_v2/val.json', 3840)
