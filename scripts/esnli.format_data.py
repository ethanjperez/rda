import math
import os
import pandas as pd
from tqdm.auto import trange


DATA_DIR = f'{os.environ["BASE_DIR"]}/data'

split2splitnames = {
    'dev': ['dev'],
    'test': ['test'],
    'train': ['train_1', 'train_2'],
}

inputtype2field = {
    'raw': 'Sentence{}',
    'marked': 'Sentence{}_marked_1',
}

inputlists = [
    'raw',
    'marked',
    'raw,explanation',
    'marked,explanation',
    'explanation',
]

split2numexplanations = {
    'train': 1,
    'dev': 3,
    'test': 3,
}


def get_field(row, field_name):
    field = row[field_name]
    if (not isinstance(field, str)) and math.isnan(field):
        return ''
    return field


def get_rationale_only(text):
    text = text.strip()
    rationale_only = ' '.join(['*' if (word.count('*') >= 2) else '_' for word in text.split()])
    assert len(set(rationale_only) - {' ', '*', '_'}) == 0, f'Found unexpected chars in rationale:\n{text}\n->\n{rationale_only}'
    return rationale_only


def mask_rationale_words(text, reverse_mask=False):
    text = text.strip()
    new_words = []
    for word in text.split():
        is_rationale_word = word.startswith('*') and word.endswith('*')
        if is_rationale_word:
            word = word[1: -1]
        mask = (not is_rationale_word) if reverse_mask else is_rationale_word
        new_words.append('_' if mask else word)
    new_text = ' '.join(new_words)
    return new_text


splits = list(sorted(split2splitnames.keys()))
for split in splits:
    print('Saving split:', split)
    splitnames = split2splitnames[split]
    dfs = [pd.read_csv(f'{DATA_DIR}/e-SNLI/esnli_{splitname}.csv') for splitname in splitnames]
    df = pd.concat(dfs, ignore_index=True)

    for inputlist in inputlists:
        save_dir = f'{DATA_DIR}/esnli.input-{inputlist}'
        print('Saving to:', save_dir)
        inputlist = inputlist.split(',')
        save_rows = ['\t'.join(['idx', 'premise', 'hypothesis', 'label'])]
        for explanation_no in range(split2numexplanations[split] if 'explanation' in inputlist else 1):
            for i in trange(len(df)):
                row = df.iloc[i]
                inputs = [[], []]
                for input_no in [1, 2]:
                    for inp in inputlist:
                        if inp == 'explanation':
                            if input_no == 1:
                                exp = row[f'Explanation_{explanation_no+1}']
                                if (not isinstance(exp, str)) and math.isnan(exp):
                                    exp = ''
                                inputs[-1].append(exp.replace('\t', ' '))
                        else:
                            field = row[inputtype2field[inp].format(str(input_no))]
                            if (not isinstance(field, str)) and math.isnan(field):
                                field = ''
                            inputs[input_no - 1].append(field.replace('\t', ' '))
                inputs = [' // '.join(inp) for inp in inputs]
                save_rows.append('\t'.join([str(len(save_rows))] + inputs + [row['gold_label']]))
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/{split}.tsv', 'w') as f:
            f.writelines('\n'.join(save_rows) + '\n')

input_types = ['markedunmasked', 'markedmasked', 'markedonly']
print('Saving additional variantions for using rationale...')
for split in splits:
    print('Saving split:', split)
    splitnames = split2splitnames[split]
    dfs = [pd.read_csv(f'{DATA_DIR}/e-SNLI/esnli_{splitname}.csv') for splitname in splitnames]
    df = pd.concat(dfs, ignore_index=True)

    for input_type in input_types:
        assert input_type in ['premise', 'hypothesis', 'markedonly', 'markedmasked', 'markedunmasked'], f'Unexpected input_type {input_type}'
        save_dir = f'{DATA_DIR}/esnli.input-{input_type}'
        print('Saving to:', save_dir)
        save_rows = ['\t'.join(['idx', 'premise', 'hypothesis', 'label'])]

        for i in trange(len(df)):
            row = df.iloc[i]
            if input_type == 'premise':
                inputs = [get_field(row, 'Sentence1'), '']
            elif input_type == 'hypothesis':
                inputs = ['', get_field(row, 'Sentence2')]
            elif input_type == 'markedonly':
                inputs = [get_rationale_only(get_field(row, 'Sentence1_marked_1')), get_rationale_only(get_field(row, 'Sentence2_marked_1'))]
            elif input_type == 'markedmasked':
                inputs = [mask_rationale_words(get_field(row, 'Sentence1_marked_1')), mask_rationale_words(get_field(row, 'Sentence2_marked_1'))]
            elif input_type == 'markedunmasked':
                inputs = [mask_rationale_words(get_field(row, 'Sentence1_marked_1'), reverse_mask=True), mask_rationale_words(get_field(row, 'Sentence2_marked_1'), reverse_mask=True)]
            inputs = [inp.replace('\t', ' ') for inp in inputs]
            save_rows.append('\t'.join([str(len(save_rows))] + inputs + [row['gold_label']]))
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/{split}.tsv', 'w') as f:
            f.writelines('\n'.join(save_rows) + '\n')
