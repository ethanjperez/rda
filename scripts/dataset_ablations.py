import argparse
import json
import os
import pandas as pd
import random
import spacy
from spacy.matcher import Matcher
from transformers.data.processors.glue import glue_processors
from tqdm.auto import tqdm


DATA_DIR = f'{os.environ["BASE_DIR"]}/data'
VERSION = 2
shuffle_rng = random.Random(42)
replace_fraction_rng = random.Random(42)
simple_save_dir_tasks = {"cola", "mrpc", "sts-b", "rte", "wnli", "qnli", "qqp", "sst-2", "anli.round1", "anli.round2", "anli.round3"}


def shuffle(tok):
    tok_list = [t.string.strip() for t in tok]
    shuffle_rng.shuffle(tok_list)
    return ' '.join(tok_list)


def mask_all(tok):
    return ('_ ' * len(tok))[:-1]


def length(tok):
    return str(len(tok))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Name of task to generate ablations for.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split of task to generate ablations for.")
    parser.add_argument("--shard_no", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    two_input_task = args.task not in {'cola', 'sst-2'}

    print('Loading data...')
    processor_name = args.task.split('.', 1)[0]
    data = getattr(glue_processors[processor_name](), f'get_{args.split}_examples')(f'{DATA_DIR}/{args.task}')
    start_idx = (args.shard_no * len(data)) // args.num_shards
    end_idx = ((args.shard_no + 1) * len(data)) // args.num_shards
    data = data[start_idx: end_idx]
    print(f'Loaded {len(data)} {args.task} examples')

    print('Loading spacy model...')
    nlp = spacy.load("en_core_web_lg")

    print('Adding function/content word matchers...')
    matcher = {}

    print('Adding POS-based Matchers...')
    pos2tags = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS'],
        'number': ['CD'],
        'wh': ['WDT', 'WP', 'WP$', 'WRB'],
        'prep': ['IN', 'TO'],
        'punct': ["''", '""', '``', '$', '-RRB-', '-LRB-', ',', ':', ';', '#', 'XX', 'SYM', "HYPH", "NFP"],  # Combining spacy POS Tag list and Sugawara et al. list
        'period': ['.'],
    }

    other_tags = {"PRP$", "-LRB-", "-RRB-", "WP$", "$", "``", "''", ",", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "XX", "_SP"}
    for pos, tags in pos2tags.items():
        for tag in tags:
            if tag in other_tags:
                other_tags.remove(tag)
    pos2tags['other'] = other_tags

    content_pos_tags = ['noun', 'verb', 'adj', 'adv', 'number']
    pos2tags['content_pos_words'] = [tag for pos in content_pos_tags for tag in pos2tags[pos]]
    function_pos_tags = {"PRP$", "-LRB-", "-RRB-", "WP$", "$", "``", "''", ",", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "SP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "XX", "_SP"}
    for pos in content_pos_tags:
        for tag in pos2tags[pos]:
            if tag in function_pos_tags:
                function_pos_tags.remove(tag)
    pos2tags['function_pos_words'] = other_tags

    for pos in ['noun', 'verb', 'adj', 'adv', 'prep', 'content_pos_words', 'function_pos_words']:
        matcher[pos] = Matcher(nlp.vocab, validate=True)
        for tag in pos2tags[pos]:
            matcher[pos].add(tag, None, [{"TAG": tag}])

    print('Adding matchers for word types...')
    gender_words = pd.read_csv(f'{os.environ["BASE_DIR"]}/gender_words_list.csv')
    wordtype2wordlist = {
        'logical_words': ['all', 'any', 'each', 'every', 'few', 'if', 'more', 'most', 'no', 'nor', 'not', "n't", 'other', 'same', 'some', 'than'],
        'causal_words': ['as', 'because', 'cause', 'reason', 'since', 'therefore', 'why'],
        'male_words': [w.lower() for w in gender_words[gender_words['gender'] == 'M']['word'].values.tolist()],
        'female_words': [w.lower() for w in gender_words[gender_words['gender'] == 'F']['word'].values.tolist()],
    }
    for wordtype, wordlist in wordtype2wordlist.items():
        print(f'{wordtype} ({len(wordlist)} words): {wordlist}')
        matcher[wordtype] = Matcher(nlp.vocab, validate=True)
        for word in wordlist:
            matcher[wordtype].add(word, None, [{"LOWER": word}])

    print('Ablating data...')
    other_transforms = [mask_all, length, shuffle]
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
        print(f'{transform}: {round(100 * frac_masked, 2)}% Masked')
        transform2frac_masked[transform] = frac_masked
    if args.split == 'test':
        transform2frac_masked_filename = f'{DATA_DIR}/{args.task}/transform2frac_masked.{args.split}.json'
        with open(transform2frac_masked_filename, 'w') as f:
            json.dump(transform2frac_masked, f)
        print('Saved Mask Fractions to:', transform2frac_masked_filename)

    for d in tqdm(data, desc='Masking Random Words...'):
        for transform, frac_masked in transform2frac_masked.items():
            transform2data[f'{transform}_fraction'].append({
                'guid': d.guid,
                'text_a': replace_word(d.tok_a, frac_masked)[0],
                'text_b': replace_word(d.tok_b, frac_masked)[0] if two_input_task else None,
                'label': d.label,
            })

    print('Saving data...')
    loadprocessor2saveprocessor = {'anli': 'anli'}  # defaults to 'nli'
    for transform, transformed_data in transform2data.items():
        assert len(data) == len(transformed_data), f'Expected len(data) == len(transformed_data) but got {len(data)} != {len(transformed_data)}'
        if args.task in simple_save_dir_tasks:
            save_dir = f'{DATA_DIR}/{args.task}.{transform}'
        else:
            save_dir = f'{DATA_DIR}/{loadprocessor2saveprocessor.get(processor_name, "nli")}.{args.task}.{transform}.v{VERSION}'
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        shard_str = f'.shard_no-{args.shard_no}.num_shards-{args.num_shards}' if args.num_shards > 1 else ''
        if processor_name == 'anli':
            key2fixedkey = {'text_a': 'context', 'text_b': 'hypothesis'}
            save_str = ''
            for td in transformed_data:
                td_fixed = {key2fixedkey.get(k, k): v for k, v in td.items()}
                td_fixed['label'] = td['label'][0]
                save_str += json.dumps(td_fixed) + '\n'
            with open(f'{save_dir}/{args.split}{shard_str}.jsonl', 'w') as f:
                f.writelines(save_str)
        else:
            transformed_lines = ['\t'.join([str(td[field]) for field in ['guid', 'text_a', 'text_b', 'label']]) for td in transformed_data]
            if args.shard_no == 0:
                transformed_lines = ['idx\ttext_a\ttext_b\tlabel'] + transformed_lines
            with open(f'{save_dir}/{args.split}{shard_str}.tsv', 'w') as f:
                f.writelines('\n'.join(transformed_lines) + '\n')


if __name__ == '__main__':
    main()

