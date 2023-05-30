import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import Pool
import multiprocessing
import random

tokenizer = AutoTokenizer.from_pretrained("../../model/bloomz-7b1-mt")

case_main_key_list = ['title', 'litigant', 'primaryFact', 'caseIntroduction', 'viewPoint', 'judgeResult'] # 主要字段

case_key_list = ['title', 'litigant', 'primaryFact', 'caseIntroduction', 'viewPoint', 'judgeResult', 'appendix', 'judgeJury'] # 所有字段

law_key_list = ['activeDate', 'content', 'issuedNo', 'lawType', 'levelType', 'lockFlag', 'publishDate', 'publisherName', 'relatedLawName', 'timelinessType', 'title', 'regionName']

new_key_list = ['title', 'content', 'region', 'industry']

qa_key_list = ['query', 'answer', 'domain', 'category'] 

viewpoint_key_list = ['title', 'author', 'publishTime', 'periodicalYear', 'wxPublicName', 'summary', 'content', 'lockFlag'] 

prosecution_document_key_list = ['title', 'primaryFact', 'viewPoint', 'badging', 'appendix', 'proDate', 'docNo', 'docType', 'proName', 'causeName']

administrative_punishment_key_list = ['title', 'content', 'pDate', 'pNo', 'regionName', 'pOrganization']


CPU_CNT = 160

def splitDoc(doc):
    return doc.split('\002')

def split2attrs(case):
    return case.split('\001')

def deleteBlank(text):
    return re.sub(r'\s+', '', text)

def cal_doc_avg_len():

    pass

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

def has_main_field(data, main_key_list):
    for key in main_key_list:
        if data[key] == '':
            return False
    return True


def cases2json(in_out_file_list):
    case_count = 0
    for (input_file, output_file) in tqdm(in_out_file_list):
        with open(input_file,'r')as f:
            # 读取整个文件
            lines = f.read()
            lines = splitDoc(lines)
        
        case_list = []
        for line in lines:
            # 正则删除文本中所有的空格
            # line = deleteBlank(line)
            # hash_code = hash(deleteBlank(line))

            attrs = split2attrs(line)
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
            case = {
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

            # 只保留高质量
            if is_hight_quality_case(case):
                case_list.append(case)

        # 所有数据写入json文件
        with open(output_file, 'w') as f:
            json.dump(case_list, f, ensure_ascii=False, indent=4)
        case_count += len(case_list)

    return case_count


# all case to json
def allCase2jsonAndSelect(fold_name, input_fold, output_fold, start_year, end_year, cpu_cnt):
    
    # 变量记录文件数
    case_count = 0

    for year in range(start_year, end_year):
        # 连接路径
        input_dir = os.path.join(input_fold, str(year))
        output_dir = os.path.join(output_fold, str(year))

        # 如果目录不存在则跳过
        if not os.path.exists(input_dir):
            continue

        # 如果没有该文件夹则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 读取目录下所有文件
        filenames = os.listdir(input_dir)

        file_path_list = []

        for filename in tqdm(filenames, desc='read case2json {} {} year file path'.format(fold_name, year)):
            # 从路径中获取文件名
            input_file = os.path.join(input_dir, filename)
            # 将文件改为json后缀
            prefix = os.path.splitext(filename)[0]

            output_file = os.path.join(output_dir, prefix+'.json')

            file_path_list.append((input_file, output_file))

        print(fold_name, year,'start')        

        # 多进程处理文件
        p_list = []
        # 每个进程需要处理的文件数
        p_size = len(file_path_list) // cpu_cnt + 1
        pool = Pool()
        for i in range(cpu_cnt):
            if i * p_size < len(file_path_list):
                p_list.append(pool.apply_async(cases2json, (file_path_list[i*p_size:(i+1)*p_size],)))
    
        pool.close()
        pool.join()
        
        print(fold_name, year,'done')

        for p in p_list:
            case_count += p.get()

    return case_count

def is_hight_quality_case(case):
    if is_verdict(case) == False:
        return False
    if is_importance_court(case) == False:
        return False
    if has_main_field(case, case_main_key_list) == False:
        return False
    return True

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
                    if is_hight_quality_case(data) == False:
                        continue 
                    cases.append(data)  
                    if len(cases) == 1000:
                        # 保存数据
                        save_data(cases, output_fold, case_count, year)
                        case_count += len(cases)
                        cases = []
        if len(cases) > 0:
            # 保存数据
            save_data(cases, output_fold, case_count, year)
            case_count += len(cases)
            cases = []

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

    if data['author'] != '':
        text = text + '作者：' + data['author'] + '\n'
    
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

def text2token(text):
    text = text + tokenizer.eos_token
    token = tokenizer.encode(text)
    return token

def process(i, datas, toTextFun):
    res = []
    for data in tqdm(datas, desc='process {} start'.format(i)):
        text = toTextFun(data)
        token = text2token(text)
        data = {'text': text, 'token': token}
        res.append(data)
    return res

# 多进程处理
def multi_process(datas, output_fold, cpu_cnt, toTextFun):

    # 每个进程处理的数据量
    p_size = len(datas) // cpu_cnt
    p_size += 1

    pool = Pool()
    p_list = []
    # 开始运行
    for i in range(cpu_cnt):
        # 最后一个进程处理剩余的数据
        if i * p_size < len(datas):
            p_list.append(pool.apply_async(process, (i, datas[i * p_size: (i + 1) * p_size], toTextFun)))
    
    pool.close()
    pool.join()

    datas = []
    batch_size = 1000
    idx = 0
    for p in tqdm(p_list, desc='get p data'):
        datas.extend(p.get())
        while len(datas) >= batch_size:
            output_file = os.path.join(output_fold, str(idx) + '.json')
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            with open(output_file, 'w', encoding='utf-8') as fp:
                fp.write('[' +',\n'.join(json.dumps(item, ensure_ascii=False) for item in datas[:batch_size]) +']\n')
            datas = datas[batch_size:]
            idx += 1

    if len(datas) > 0:
        output_file = os.path.join(output_fold, str(idx) + '.json')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        with open(output_file, 'w', encoding='utf-8') as fp:
            fp.write('[' +',\n'.join(json.dumps(item, ensure_ascii=False) for item in datas[:]) +']\n')
        datas = []
        idx += 1
    


def case2TextAndToken(fold_name, input_fold, output_fold, start_year, end_year):
    # 将案例转化为文本
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
        
        multi_process(json_datas, os.path.join(output_fold, str(year)), CPU_CNT, case2text)



def save_data(cases, output_fold, idx, year, indent=4):
    # 保存数据
    output_file = os.path.join(output_fold, str(year), str(idx) + '.json')
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        json.dump(cases, f, ensure_ascii=False, indent=indent)


def zh_cnt(text):
    count = 0
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            count += 1
    return count

def getTraintoken(input_file_list, output_fold):
    tokens_list = []
    token_queue = []
    max_len = 2048
    batch_size = 100000 # 处理一次
    # 文件大小达到一定程度，停止
    cnt=0
    for input_file in tqdm(input_file_list, desc='to token'):
        with open(input_file, 'r')as f:
            # 读取整个文件
            json_datas = json.load(f)            
            for data in json_datas:
                # 判断中文比例
                count = zh_cnt(data['text'])
                # 如果汉字比例小于0.5则跳过
                if count / len(data['text']) < 0.5:
                    continue

                tokens_list.append(data['token'])

        if len(tokens_list) >= batch_size: # 处理一次
            
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
         

def data2jsonAndSelect(input_fold, output_fold, key_list, title_name, main_key_list, min_len):
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
                
                # 检查doc
                if is_hight_quality_doc(data, title_name, main_key_list, min_len):
                    doc_list.append(data)

            # 所有数据写入json文件
            with open(output_file, 'w') as f:
                json.dump(doc_list, f, ensure_ascii=False, indent=4)

def select_data(input_fold, output_fold, title_name, main_key_list, min_len):
    filenames = os.listdir(input_fold)
    datas = []
    for filename in tqdm(filenames, desc='select data'):
        # 从路径中获取文件名
        input_file = os.path.join(input_fold, filename)
        with open(input_file,'r')as f:
            # 读取整个文件
            json_datas = json.load(f)
            for data in json_datas:
                if is_hight_quality_doc(data, title_name, main_key_list, min_len) == False:
                    continue
                datas.append(data)  
    # 每1000个案例保存一次
    for i in range(0, len(datas), 1000):
        output_file = os.path.join(output_fold, str(i) + '.json')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w') as f:
            json.dump(datas[i:i+1000], f, ensure_ascii=False, indent=4)

def is_hight_quality_doc(doc, title_name, main_key_list, min_len):
    if doc[title_name] == '':
        return False

    for main_key in main_key_list:
        # 删除网页标签
        doc[main_key] = re.sub(r'<[^>]+>', '', doc[main_key])
        if len(doc[main_key]) < min_len: # 如果content长度小于100则跳过
            return False
        # 统计汉字的个数
        count = 0
        for c in doc[main_key]:
            if '\u4e00' <= c <= '\u9fff':
                count += 1
        # 如果汉字比例小于0.5则跳过
        if count / len(doc[main_key]) < 0.5:
            return False
    return True




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
    
    multi_process(json_datas, output_fold, CPU_CNT, toTextFun)


def get_textAndToken_file_list(raw_dir, data_type_list, want_data_type_size):

    paths = {}

    def gci(filepath, data_type):
        files = os.listdir(filepath)
        for fi in files:
            fi_d = os.path.join(filepath, fi)            
            if os.path.isdir(fi_d):
                gci(fi_d, data_type)                  
            else:
                paths[data_type].append(os.path.join(filepath, fi_d))

    for data_type in data_type_list:
        paths[data_type] = []
        if data_type == 'case':
            for c in ['administrative_case2', 'civil_case2', 'criminal_cases2', 'execution_of_cases2', 'national_compensation_cases2']:
                input_fold = os.path.join(raw_dir, data_type, c, 'textAndtoken')
                gci(input_fold, data_type)
        else:
            input_fold = os.path.join(raw_dir, data_type, 'textAndtoken')
            gci(input_fold, data_type)

    for data_type in data_type_list:
        print(data_type, len(paths[data_type]))
    
    # 获取总的文件大小
    data_type_size = {}

    for data_type in data_type_list:
        data_type_size[data_type] = 0
        for path in paths[data_type]:
            data_type_size[data_type] += os.path.getsize(path)
    
    print('数据类别大小')
    for key in data_type_list:
        print(key, data_type_size[key]/1024/1024/1024, 'GB')
    
    # 计算比例
    total_size = 0
    for key in data_type_list:
        total_size += data_type_size[key]
    
    # 数据比例
    print('数据比例')
    for key in data_type_list: # 计算比例
        data_type_size[key] = data_type_size[key] / total_size
        print(key, data_type_size[key])

    real_data_type_size={}
    # 随机选择, 按比例划分
    for data_type in want_data_type_size:
        random.shuffle(paths[data_type])
        tmp_list = []
        tmp_size = 0
        for path in paths[data_type]:
            tmp_size += os.path.getsize(path)
            tmp_list.append(path)
            if tmp_size >= want_data_type_size[data_type]:
                break
        real_data_type_size[data_type] = tmp_size
        paths[data_type] = tmp_list[:]
    # 输出大小
    print('选取数据真实大小')
    all_size = 0
    for key in data_type_list:
        print(key, real_data_type_size[key]/1024/1024/1024, 'GB')
        all_size += real_data_type_size[key]
    print('总大小', all_size/1024/1024/1024, 'GB')
    return paths    

def genTrainData(input_fold, output_fold, train_file, valid_file, valid_size):

    train_file = os.path.join(output_fold, train_file)

    valid_file = os.path.join(output_fold, valid_file)

    # 列出所有文件
    filenames = os.listdir(input_fold)
    # 从文件中均匀取数据放到验证集中
    pre_file_valid_size = valid_size // len(filenames)

    first_file_valid_size = pre_file_valid_size + valid_size % len(filenames)

    with open(train_file, 'w') as tf:
        with open(valid_file, 'w') as vf:
            for i, filename in enumerate(tqdm(filenames, desc='gen train data')):
                input_file = os.path.join(input_fold, filename)
                with open(input_file, 'r') as f:
                    # 读取整个文件
                    datas = f.readlines()
                    if i == 0:
                        for data in datas[:first_file_valid_size]:
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            vf.write(data)
                        for data in datas[first_file_valid_size:]:
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
                        for data in datas[pre_file_valid_size:]:
                            json_data = json.loads(data)
                            token = json_data['token']
                            assert len(token) == 2048
                            tf.write(data)

def del_case(input_dir, output_dir):
    input_dir = os.path.join(input_dir, 'case')
    output_dir = os.path.join(output_dir, 'case')
    case_fold_list = [
        'civil_case2',
        'administrative_case2', 
        'criminal_cases2',
        'execution_of_cases2',
        'national_compensation_cases2'
    ]

    case_count = 0
    for fold in case_fold_list:
        input_data_fold = os.path.join(input_dir, fold)
        # json_fold = os.path.join(output_dir, fold, 'json') # 转json
        selected_fold = os.path.join(output_dir, fold, 'select') # 选取
        textAndtoken_fold = os.path.join(output_dir, fold, 'textAndtoken') # textAndtoken
        case_count += allCase2jsonAndSelect(fold, input_data_fold, selected_fold, 2020, 2024, CPU_CNT)
        # select_case(fold, json_fold, selected_fold, 2020, 2024)
        case2TextAndToken(fold, selected_fold, textAndtoken_fold, 2020, 2024)

    print(case_count)

def del_other(input_dir, output_dir, d_type, min_len, toTextFun, key_list, title_name, main_key_list):
    input_dir = os.path.join(input_dir, d_type)
    output_dir = os.path.join(output_dir, d_type)

    select_fold = os.path.join(output_dir, 'select')
    textAndtoken_fold = os.path.join(output_dir, 'textAndtoken') # textAndtoken
    # 如果没有目录，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not os.path.exists(json_fold):
    #     os.makedirs(json_fold)
    if not os.path.exists(select_fold):
        os.makedirs(select_fold)
    if not os.path.exists(textAndtoken_fold):
        os.makedirs(textAndtoken_fold)
    data2jsonAndSelect(input_dir, select_fold, key_list, title_name, main_key_list, min_len)
    # select_data(json_fold, select_fold, 'title', ['content'], 100)
    ToTextAndToken(select_fold, textAndtoken_fold, toTextFun)


if __name__ == '__main__': 

    input_dir = '../../data/deli_data/'   
    # output_dir = '../../data/processed_deli_data/'
    output_dir = '/data1/wenjiabao/HanFei/data/processed_deli_data/'
    
    # 处理案例
    # del_case(input_dir, output_dir)

    # 处理laws
    # del_other(input_dir, output_dir, 'laws', 100, law2text, law_key_list, 'title', ['content'])

    # 处理news
    # del_other(input_dir, output_dir, 'news', 100, new2text, new_key_list, 'title', ['content'])

    # 处理prosecution_document
    # del_other(input_dir, output_dir, 'prosecution_document', 50, prosecution_document2text, prosecution_document_key_list, 'title', ['primaryFact', 'viewPoint'])

    # 处理qa
    # del_other(input_dir, output_dir, 'qa', 30, qa2text, qa_key_list, 'query', ['answer'])

    # 处理viewpoint
    # del_other(input_dir, output_dir, 'viewpoint', 100, viewpoint2text, viewpoint_key_list, 'title', ['content'])

    # 处理administrative_punishment
    # del_other(input_dir, output_dir, 'administrative_punishment', 50, administrative_punishment2text, administrative_punishment_key_list, 'title', ['content'])

    # textAndtoken_fold_list = [
    #     '/mntcephfs/data/med/wjb/LawChatGPT/data/viewpoint/textAndtoken',
    # ]

    d_type_list = ['case', 'laws', 'news', 'prosecution_document', 'qa', 'viewpoint', 'administrative_punishment']

    # 想要取的每个类别大小
    want_data_type_size = {
        'case': 15 * 1024 * 1024 * 1024,
        'laws': 30 * 1024 * 1024 * 1024,
        'news': 30 * 1024 * 1024 * 1024,
        'prosecution_document': 30 * 1024 * 1024 * 1024,
        'qa': 30 * 1024 * 1024 * 1024,
        'viewpoint': 30 * 1024 * 1024 * 1024,
        'administrative_punishment': 30 * 1024 * 1024 * 1024
    }
    
    all_textAndToken_file_dict = get_textAndToken_file_list(output_dir, d_type_list, want_data_type_size)


    all_textAndToken_file_list = []
    for key in all_textAndToken_file_dict:
        all_textAndToken_file_list += all_textAndToken_file_dict[key]

    output_fold = '/data1/wenjiabao/HanFei/data/pretrain/train_token'

    # # shuffle所有文件
    random.shuffle(all_textAndToken_file_list)

    getTraintoken(all_textAndToken_file_list, output_fold)

    input_fold = '/data1/wenjiabao/HanFei/data/pretrain/train_token'

    output_fold = '/data1/wenjiabao/HanFei/data/pretrain'

    genTrainData(input_fold, output_fold, 'train.json', 'valid.json', 3840)