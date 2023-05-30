import random

import jsonlines as jl
import json
import pandas as pd
import random

jlfile = '/fastdata/act21hz/general_pretrain/resources/PAQ.metadata.jsonl'
converted_file = '/fastdata/act21hz/general_pretrain/resources/PAQ.QA.jsonl'
df_file = '/fastdata/act21hz/general_pretrain/resources/PAQ.QA.tsv'
df = pd.DataFrame(columns=['question', 'answer', 'passage_id'])
results = []
for ln, line in enumerate(jl.open(jlfile)):
    data = line
    # data = {'question': data['question'], 'answer': data['answer'],
    #         'passage_id': [ans['passage_id'] for ans in data['answers']]}
    results.append(
        [data['question'], random.choice(data['answer']), [int(ans['passage_id']) for ans in data['answers']]])
    print(f'Loaded {ln + 1} Items from {jlfile}', flush=True) if ln % 1000000 == 0 else None
df = pd.DataFrame(results, columns=['question', 'answer', 'passage_id'])
df.to_csv(df_file, sep='\t')
