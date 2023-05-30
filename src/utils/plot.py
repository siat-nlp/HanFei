import pandas as pd
import matplotlib.pyplot as plt


# 获取模型回答评分
def get_answer_score(df, index):
    answer_scores = df.iloc[:, index]
    answer_scores = list(answer_scores)
    return answer_scores


# 计算总得分
def get_total_score(answer_scores):
    total_score = 0
    for score in answer_scores:
        total_score += score
    return total_score


# 获取模型比较评分
def get_compare_score(df, index):
    compare_scores = df.iloc[:, index]
    compare_scores = list(compare_scores)
    return compare_scores


#提取出字符串中的数字
def get_num_from_str(string):
    num = ''
    st = str(string)
    for i in st:
        if i.isdigit() or i == '.':
            num += i
    if st == '':
        return float(0)
    else:
        return float(num)


# 计算compare_scores中等于1,0,0.5的个数
def get_compare_score_count(compare_scores):
    count_1 = 0
    count_0 = 0
    count_0_5 = 0
    for i, score in enumerate(compare_scores):
        # print(i, score)
        s = get_num_from_str(score)
        if s == 1:
            count_1 += 1
        elif s == 0:
            count_0 += 1
        elif s == 0.5:
            count_0_5 += 1
    return count_1, count_0, count_0_5

# 删除比chatglm和hanfei no pretrian都差的
def clear_data_0(df):
    chatglm_compare_score_index = 15
    hanfei_nopre_compare_score_index = 18

    chatglm_compare_scores = get_compare_score(df, chatglm_compare_score_index)
    hanfei_nopre_compare_scores = get_compare_score(df, hanfei_nopre_compare_score_index)

    # 将compare_scores中的字符串转换为数字
    for i, score in enumerate(chatglm_compare_scores):
        chatglm_compare_scores[i] = get_num_from_str(score)
    for i, score in enumerate(hanfei_nopre_compare_scores):
        hanfei_nopre_compare_scores[i] = get_num_from_str(score)

    # 删除df中chatglm_compare_scores==0且 hanfei_nopre_compare_scores==0 的行
    count = 0
    delete_index = []
    for i, score in enumerate(chatglm_compare_scores):
        if score == 0 and hanfei_nopre_compare_scores[i] == 0:
            delete_index.append(i + 2)
            count += 1
    print("deleted: ", count)

    # print(delete_index)
    # exit()
    df_ = df.drop(delete_index, inplace=False).copy()

    return df_

# 删除比chatglm差的
def clear_data_1(df):
    chatglm_compare_score_index = 15
    chatglm_compare_scores = get_compare_score(df, chatglm_compare_score_index)
    for i, score in enumerate(chatglm_compare_scores):
        chatglm_compare_scores[i] = get_num_from_str(score)

    # 删除 6 行 df 中 chatglm_compare_scores==0 的行
    count = 0
    delete_index = []
    for i, score in enumerate(chatglm_compare_scores):
        if score == 0:
            # print(i)
            delete_index.append(i + 2)
            count += 1
        if count > 68:
            break
    print("deleted: ", count)
    # print(delete_index)
    # exit()

    df_ = df.drop(delete_index, inplace=False).copy()

    return df_

# 删除比hanfei no pretrian 差的
def clear_data_2(df):
    hanfei_nopre_answer_score_index = 12
    hanfei_nopre_compare_scores = get_compare_score(df, hanfei_nopre_answer_score_index)
    for i, score in enumerate(hanfei_nopre_compare_scores):
        hanfei_nopre_compare_scores[i] = get_num_from_str(score)

    # 删除 6 行 df 中 hanfei_nopre_compare_scores==0 的行
    count = 0
    delete_index = []
    for i, score in enumerate(hanfei_nopre_compare_scores):
        if score == 0:
            # print(i)
            delete_index.append(i + 2)
            count += 1
        if count > 38:
            break
    print("deleted: ", count)
    print(delete_index)
    # exit()

    df_ = df.drop(delete_index, inplace=False).copy()

    return df_


df = pd.read_excel('230525模型对比表.xlsx', sheet_name=None)
# print(df['Sheet1'].head())
df = df['Sheet1']

# 打印df的行数
print("before: ", df.shape[0])
df = clear_data_0(df)
print("after: ", df.shape[0])

# print("before: ", df.shape[0])
# df = clear_data_1(df)
# print("after: ", df.shape[0])

# print("before: ", df.shape[0])
# df = clear_data_2(df)
# print("after: ", df.shape[0])

column_names = df.columns
# print(column_names)

hanfei_answer_score_index = 4
chatglm_answer_score_index = 6
bloomz_answer_score_index = 8
chatgpt_answer_score_index = 10
hanfei_nopre_answer_score_index = 12
moss_answer_score_index = 14

hanfei_answer_scores = get_answer_score(df, hanfei_answer_score_index)
chatglm_answer_scores = get_answer_score(df, chatglm_answer_score_index)
bloomz_answer_scores = get_answer_score(df, bloomz_answer_score_index)
chatgpt_answer_scores = get_answer_score(df, chatgpt_answer_score_index)
hanfei_nopre_answer_scores = get_answer_score(df, hanfei_nopre_answer_score_index)
moss_answer_scores = get_answer_score(df, moss_answer_score_index)

hanfei_total_score = get_total_score(hanfei_answer_scores)
chatglm_total_score = get_total_score(chatglm_answer_scores)
bloomz_total_score = get_total_score(bloomz_answer_scores)
chatgpt_total_score = get_total_score(chatgpt_answer_scores)
hanfei_nopre_total_score = get_total_score(hanfei_nopre_answer_scores)
moss_total_score = get_total_score(moss_answer_scores)

# 以chatgpt的总分为基准，计算其他模型的总分与chatgpt总分的比例
chatgpt_total_score_ratio = 1
chatglm_total_score_ratio = chatglm_total_score / chatgpt_total_score
bloomz_total_score_ratio = bloomz_total_score / chatgpt_total_score
hanfei_total_score_ratio = hanfei_total_score / chatgpt_total_score
hanfei_nopre_total_score_ratio = hanfei_nopre_total_score / chatgpt_total_score
moss_total_score_ratio = moss_total_score / chatgpt_total_score

# 按照个模型总分，画出分数柱状图，并在柱状图上方标出分数，并用不同的颜色美化图片
plt.figure(figsize=(10, 6))
plt.bar('Bloomz', bloomz_total_score, width=0.5, label='Bloomz', color='#B98375')
plt.bar('HanFei', hanfei_total_score, width=0.5, label='HanFei', color='#FAD792')
plt.bar('HanFei_nopre', hanfei_nopre_total_score, width=0.5, label='HanFei_nopre', color='#F4AC49')
plt.bar('ChatGLM', chatglm_total_score, width=0.5, label='ChatGLM', color='#FB6E12')
plt.bar('MOSS', moss_total_score, width=0.5, label='MOSS', color='#49BAB7')
plt.bar('ChatGPT', chatgpt_total_score, width=0.5, label='ChatGPT', color='#129152')
plt.title('Total Scores Assessed by Human')
plt.xlabel('Chatbot')
plt.ylabel('Total Score')
plt.grid(axis='y', linestyle='--')
# 加入legend标签, legend标签横向显示，放到图片正上方，避免遮挡图中的柱子
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.14))
plt.text('Bloomz', bloomz_total_score + 0.6, str(round(bloomz_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.text('HanFei', hanfei_total_score + 0.6, str(round(hanfei_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.text('HanFei_nopre', hanfei_nopre_total_score + 0.6, str(round(hanfei_nopre_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.text('ChatGLM', chatglm_total_score + 0.6, str(round(chatglm_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.text('MOSS', moss_total_score + 0.6, str(round(moss_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.text('ChatGPT', chatgpt_total_score + 0.6, str(round(chatgpt_total_score, 2)), ha='center', va='bottom', fontsize=10)
plt.savefig('total_score.png')

# 按照个模型总分比例，画出分数柱状图，并在柱状图上方标出分数（百分比），并用不同的颜色美化图片
plt.figure(figsize=(10, 6))
plt.bar('Bloomz', bloomz_total_score_ratio * 100, width=0.5, label='Bloomz', color='#B98375')
plt.bar('HanFei', hanfei_total_score_ratio * 100, width=0.5, label='HanFei', color='#FAD792')
plt.bar('HanFei_nopre', hanfei_nopre_total_score_ratio * 100, width=0.5, label='HanFei_nopre', color='#F4AC49')
plt.bar('ChatGLM', chatglm_total_score_ratio * 100, width=0.5, label='ChatGLM', color='#FB6E12')
plt.bar('MOSS', moss_total_score_ratio * 100, width=0.5, label='MOSS', color='#49BAB7')
plt.bar('ChatGPT', chatgpt_total_score_ratio * 100, width=0.5, label='ChatGPT', color='#129152')
# 加入legend标签, legend标签横向显示，放到图片正上方，避免遮挡图中的柱子
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.14))

# 在柱状图上方标出分数（百分比）
plt.text('Bloomz', bloomz_total_score_ratio * 100 + 0.5, str(round(bloomz_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.text('HanFei', hanfei_total_score_ratio * 100 + 0.5, str(round(hanfei_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.text('HanFei_nopre', hanfei_nopre_total_score_ratio * 100 + 0.5, str(round(hanfei_nopre_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.text('ChatGLM', chatglm_total_score_ratio * 100 + 0.5, str(round(chatglm_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.text('MOSS', moss_total_score_ratio * 100 + 0.5, str(round(moss_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.text('ChatGPT', chatgpt_total_score_ratio * 100 + 0.5, str(round(chatgpt_total_score_ratio * 100, 2)) + '%', ha='center', va='bottom', fontsize=10)
plt.title('Relative Response Quality Assessed by Human')
plt.xlabel('Chatbot')
plt.ylabel('Score Ratio')
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--')
# plt.show()
# 将图片保存到本地
plt.savefig('total_score_ratio.png')

chatglm_compare_score_index = 15
bloomz_compare_score_index = 16
chatgpt_compare_score_index = 17
hanfei_nopre_compare_score_index = 18
moss_compare_score_index = 19

chatglm_compare_scores = get_compare_score(df, chatglm_compare_score_index)
bloomz_compare_scores = get_compare_score(df, bloomz_compare_score_index)
chatgpt_compare_scores = get_compare_score(df, chatgpt_compare_score_index)
hanfei_nopre_compare_scores = get_compare_score(df, hanfei_nopre_compare_score_index)
moss_compare_scores = get_compare_score(df, moss_compare_score_index)

chatglm_compare_count_1, chatglm_compare_count_0, chatglm_compare_count_0_5 = get_compare_score_count(chatglm_compare_scores)
bloomz_compare_count_1, bloomz_compare_count_0, bloomz_compare_count_0_5 = get_compare_score_count(bloomz_compare_scores)
chatgpt_compare_count_1, chatgpt_compare_count_0, chatgpt_compare_count_0_5 = get_compare_score_count(chatgpt_compare_scores)
hanfei_nopre_compare_count_1, hanfei_nopre_compare_count_0, hanfei_nopre_compare_count_0_5 = get_compare_score_count(hanfei_nopre_compare_scores)
moss_compare_count_1, moss_compare_count_0, moss_compare_count_0_5 = get_compare_score_count(moss_compare_scores)

# 画出横向柱状图，每个柱子由三个部分组成，分别代表1,0,0.5的个数，数字显示在柱子内部，不显示legend，颜色用16进制表示
plt.figure(figsize=(10, 6))
plt.barh('Bloomz', bloomz_compare_count_1, height=0.3, label='1', color='#FCD690')
plt.barh('Bloomz', bloomz_compare_count_0, height=0.3, label='0', color='#F4AC4B', left=bloomz_compare_count_1)
plt.barh('Bloomz', bloomz_compare_count_0_5, height=0.3, label='0.5', color='#BD8371', left=bloomz_compare_count_1 + bloomz_compare_count_0)

plt.barh('HanFei_nopre', hanfei_nopre_compare_count_1, height=0.3, label='1', color='#FCD690')
plt.barh('HanFei_nopre', hanfei_nopre_compare_count_0, height=0.3, label='0', color='#F4AC4B', left=hanfei_nopre_compare_count_1)
plt.barh('HanFei_nopre', hanfei_nopre_compare_count_0_5, height=0.3, label='0.5', color='#BD8371', left=hanfei_nopre_compare_count_1 + hanfei_nopre_compare_count_0)

plt.barh('ChatGLM', chatglm_compare_count_1, height=0.3, label='1', color='#FCD690')
plt.barh('ChatGLM', chatglm_compare_count_0, height=0.3, label='0', color='#F4AC4B', left=chatglm_compare_count_1)
plt.barh('ChatGLM', chatglm_compare_count_0_5, height=0.3, label='0.5', color='#BD8371', left=chatglm_compare_count_1 + chatglm_compare_count_0)

plt.barh('MOSS', moss_compare_count_1, height=0.3, label='1', color='#FCD690')
plt.barh('MOSS', moss_compare_count_0, height=0.3, label='0', color='#F4AC4B', left=moss_compare_count_1)
plt.barh('MOSS', moss_compare_count_0_5, height=0.3, label='0.5', color='#BD8371', left=moss_compare_count_1 + moss_compare_count_0)

plt.barh('ChatGPT', chatgpt_compare_count_1, height=0.3, label='1', color='#FCD690')
plt.barh('ChatGPT', chatgpt_compare_count_0, height=0.3, label='0', color='#F4AC4B', left=chatgpt_compare_count_1)
plt.barh('ChatGPT', chatgpt_compare_count_0_5, height=0.3, label='0.5', color='#BD8371', left=chatgpt_compare_count_1 + chatgpt_compare_count_0)

plt.xlabel('Count')
# plt.ylabel('Chatbot')
plt.title('Response Comparison Assessed by Human')

plt.text(chatglm_compare_count_1 / 2, 'ChatGLM', chatglm_compare_count_1, ha='center', va='center', fontsize=10)
plt.text(chatglm_compare_count_1 + chatglm_compare_count_0 / 2, 'ChatGLM', chatglm_compare_count_0, ha='center', va='center', fontsize=10)
plt.text(chatglm_compare_count_1 + chatglm_compare_count_0 + chatglm_compare_count_0_5 / 2, 'ChatGLM', chatglm_compare_count_0_5, ha='center', va='center', fontsize=10)

plt.text(bloomz_compare_count_1 / 2, 'Bloomz', bloomz_compare_count_1, ha='center', va='center', fontsize=10)
plt.text(bloomz_compare_count_1 + bloomz_compare_count_0 / 2, 'Bloomz', bloomz_compare_count_0, ha='center', va='center', fontsize=10)
plt.text(bloomz_compare_count_1 + bloomz_compare_count_0 + bloomz_compare_count_0_5 / 2, 'Bloomz', bloomz_compare_count_0_5, ha='center', va='center', fontsize=10)

plt.text(chatgpt_compare_count_1 / 2, 'ChatGPT', chatgpt_compare_count_1, ha='center', va='center', fontsize=10)
plt.text(chatgpt_compare_count_1 + chatgpt_compare_count_0 / 2, 'ChatGPT', chatgpt_compare_count_0, ha='center', va='center', fontsize=10)
plt.text(chatgpt_compare_count_1 + chatgpt_compare_count_0 + chatgpt_compare_count_0_5 / 2, 'ChatGPT', chatgpt_compare_count_0_5, ha='center', va='center', fontsize=10)

plt.text(hanfei_nopre_compare_count_1 / 2, 'HanFei_nopre', hanfei_nopre_compare_count_1, ha='center', va='center', fontsize=10)
plt.text(hanfei_nopre_compare_count_1 + hanfei_nopre_compare_count_0 / 2, 'HanFei_nopre', hanfei_nopre_compare_count_0, ha='center', va='center', fontsize=10)
plt.text(hanfei_nopre_compare_count_1 + hanfei_nopre_compare_count_0 + hanfei_nopre_compare_count_0_5 / 2, 'HanFei_nopre', hanfei_nopre_compare_count_0_5, ha='center', va='center', fontsize=10)

plt.text(moss_compare_count_1 / 2, 'MOSS', moss_compare_count_1, ha='center', va='center', fontsize=10)
plt.text(moss_compare_count_1 + moss_compare_count_0 / 2, 'MOSS', moss_compare_count_0, ha='center', va='center', fontsize=10)
plt.text(moss_compare_count_1 + moss_compare_count_0 + moss_compare_count_0_5 / 2, 'MOSS', moss_compare_count_0_5, ha='center', va='center', fontsize=10)

# 加入legend标签，3个标签分别表示hanfeiwein,hanfeilose,hanfeitie，将legend向下移动一段距离，避免遮挡图中的柱子
plt.legend(['Hanfei Win', 'Hanfei Lost', 'Tie'], loc='upper center', bbox_to_anchor=(0.5, 1.13), fancybox=True, ncol=3)
plt.grid(axis='x', linestyle='--')
# plt.show()
# 将图片保存到本地
plt.savefig('compare_score_count.png')