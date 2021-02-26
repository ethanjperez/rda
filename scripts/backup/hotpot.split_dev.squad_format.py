import json
import os


DATA_DIR = f'{os.environ["BASE_DIR"]}/data'

with open(f'{DATA_DIR}/hotpot-orig/dev1.qids.txt') as f:
    qids1 = {line.strip() for line in f}

data_folders = [
    f'hotpot.context-long.subqs-{model_no}.num_subas-{num_subas}.shuffled-1'
    for model_no in [20604919, 21979200, 22008128, 22026601, 22165087, 7]
    for num_subas in [1, 2]
]

for data_folder in data_folders:
    with open(f'{DATA_DIR}/{data_folder}/dev.json') as f:
        data = json.load(f)

    with open(f'{DATA_DIR}/{data_folder}/dev1.json', 'w') as f:
        json.dump({
            'data': [d for d in data['data'] if d['paragraphs'][0]['_id'] in qids1],
            'version': data['version']
        }, f)

    with open(f'{DATA_DIR}/{data_folder}/dev2.json', 'w') as f:
        json.dump({
            'data': [d for d in data['data'] if d['paragraphs'][0]['_id'] not in qids1],
            'version': data['version']
        }, f)
