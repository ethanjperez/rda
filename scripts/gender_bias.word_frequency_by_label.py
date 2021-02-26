import argparse
import os
import pandas as pd
import random
import spacy
from spacy.matcher import Matcher
from transformers.data.processors.glue import glue_processors
from tqdm.auto import tqdm


simple_save_dir_tasks = {"cola", "mrpc", "sts-b", "rte", "wnli", "qnli", "qqp", "sst-2", "anli.round1", "anli.round2", "anli.round3"}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=simple_save_dir_tasks, help="Name of task on which to examine word type frequencies by class label")
parser.add_argument("--split", type=str, default='train', choices=['train', 'dev', 'test'], help="Split of dataset you would like to examine")
args = parser.parse_args()
args.shard_no = 0
args.num_shards = 1
replace_fraction_rng = random.Random(42)


def replace_word(tok, matcher_func_or_frac, replacement='_'):
    text = ''
    buffer_start = 0
    if isinstance(matcher_func_or_frac, float) or isinstance(matcher_func_or_frac, int):
        match_starts = [i for i in range(len(tok)) if replace_fraction_rng.random() < matcher_func_or_frac]
    else:
        match_starts = [match_start for _, match_start, _ in matcher_func_or_frac(tok)]
    for match_start in match_starts:
        if match_start > buffer_start:  # If we've skipped over some tokens, let's add those in (with trailing whitespace if available)
            text += tok[buffer_start: match_start].text + tok[match_start - 1].whitespace_
        if len(tok[match_start].text.strip()) == 0:
            text += tok[match_start].string
        else:
            text += replacement + tok[match_start].whitespace_  # Replace token, with trailing whitespace if available
        buffer_start = match_start + 1
    if len(tok) > 0:
        text += tok[buffer_start:].text
    return text.strip(), len(match_starts)


two_input_task = args.task not in {'cola', 'sst-2'}

print('Loading data...')
processor_name = args.task.split('.', 1)[0]
data = getattr(glue_processors[processor_name](), f'get_{args.split}_examples')(f'{os.environ["BASE_DIR"]}/data/{args.task}')
start_idx = (args.shard_no * len(data)) // args.num_shards
end_idx = ((args.shard_no + 1) * len(data)) // args.num_shards
data = data[start_idx: end_idx]
print(f'Loaded {len(data)} {args.task} examples')

nlp = spacy.load("en_core_web_lg")

matcher = {}
gender_words = pd.read_csv(f'{os.environ["BASE_DIR"]}/gender_words_list.csv')
wordtype2wordlist = {
    'male_words': [w.lower() for w in gender_words[gender_words['gender'] == 'M']['word'].values.tolist()],
    'female_words': [w.lower() for w in gender_words[gender_words['gender'] == 'F']['word'].values.tolist()],
}
for wordtype, wordlist in wordtype2wordlist.items():
    matcher[wordtype] = Matcher(nlp.vocab, validate=True)
    for word in wordlist:
        matcher[wordtype].add(word, None, [{"LOWER": word}])

print('Ablating data...')
other_transforms = []
transform2data = {f'mask_{matcher_name}': [] for matcher_name in matcher}
for matcher_name in matcher:
    transform2data[f'mask_{matcher_name}_fraction'] = []
for transform in other_transforms:
    transform2data[transform.__name__] = []
transform2num_masked = {f'mask_{matcher_name}': 0 for matcher_name in matcher}

for d in tqdm(data):
    d.tok_a = nlp(d.text_a.strip())
    if two_input_task:
        d.tok_b = nlp(d.text_b.strip())
    for matcher_name, matcher_func in matcher.items():
        transform = f'mask_{matcher_name}'
        masked_text_a, num_masked_a = replace_word(d.tok_a, matcher_func)
        masked_text_b, num_masked_b = replace_word(d.tok_b, matcher_func) if two_input_task else (None, 0)
        transform2num_masked[transform] += num_masked_a + num_masked_b
        transform2data[transform].append({
            'guid': d.guid,
            'text_a': masked_text_a,
            'text_b': masked_text_b,
            'label': d.label,
            'num_masked_a': num_masked_a,
            'num_masked_b': num_masked_b,
        })

    for transform in other_transforms:
        transform2data[transform.__name__].append({
            'guid': d.guid,
            'text_a': transform(d.tok_a),
            'text_b': transform(d.tok_b) if two_input_task else None,
            'label': d.label,
        })

print('Calculating Mask Fractions...')
num_tokens = sum((len(d.tok_a) + len(d.tok_b)) if two_input_task else len(d.tok_a) for d in data)
transform2frac_masked = {}
for transform, num_masked in transform2num_masked.items():
    frac_masked = num_masked / num_tokens
    print(f'\t{transform}: {round(100 * frac_masked, 2)}% Masked')
    transform2frac_masked[transform] = frac_masked

for d in tqdm(data, desc='Masking Random Words...'):
    for transform, frac_masked in transform2frac_masked.items():
        transform2data[f'{transform}_fraction'].append({
            'guid': d.guid,
            'text_a': replace_word(d.tok_a, frac_masked)[0],
            'text_b': replace_word(d.tok_b, frac_masked)[0] if two_input_task else None,
            'label': d.label,
        })

for transform in transform2data.keys():
    if 'fraction' in transform:
        continue
    print(transform)
    label2nummaskeda = {}
    label2numtokensa = {}
    if two_input_task:
        label2nummaskedb = {}
        label2numtokensb = {}
    for d, d_info in zip(data, transform2data[transform]):
        l = d_info['label']
        label2nummaskeda[l] = label2nummaskeda.get(l, 0) + d_info['num_masked_a']
        label2numtokensa[l] = label2numtokensa.get(l, 0) + len(d.tok_a)
        if two_input_task:
            label2nummaskedb[l] = label2nummaskedb.get(l, 0) + d_info['num_masked_b']
            label2numtokensb[l] = label2numtokensb.get(l, 0) + len(d.tok_b)
    for label, nummaskeda in label2nummaskeda.items():
        print(f'\t{label}: Sent 1: {round(100. * nummaskeda / label2numtokensa[label], 4)}%')
        if two_input_task:
            print(f'\t{label}: Sent 2: {round(100. * label2nummaskedb[label] / label2numtokensb[label], 4)}%')
