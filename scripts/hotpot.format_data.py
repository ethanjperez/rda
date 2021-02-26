import argparse
import json
import numpy as np
import os
from pprint import pprint
import random
from tqdm.auto import tqdm


DATA_DIR = f'{os.environ["BASE_DIR"]}/UnsupervisedDecomposition/XLM/data'


def convert_example(example, num_subanswers, use_all_answers=False, title2sa_nos=None, subqs=None):
    answer = example['answer'].strip()
    new_example = {
        "title": example['question'],
        'paragraphs': [{
            # HotpotQA info
            "_id": example['_id'],
            "type": example['type'],
            "level": example['level'],
            "supporting_facts": [],  # To add
            # SQuAD info
            "context": 'yes no',  # To add
            "qas": [
                {
                    "question": example['question'],
                    "id": example['_id'],
                    "answers": []  # To add
                }
            ]
        }]
    }

    answer_starts = []
    answer_starts_sf_idxs = []
    span_q = True
    if answer.lower() in ['yes', 'no']:
        answer_starts.append(new_example['paragraphs'][0]['context'].index(example['answer'].lower().strip()))
        answer_starts_sf_idxs.append(-1)
        span_q = False

    sf_titles = []
    for title, sents in example['supporting_facts']:
        if title not in sf_titles:
            sf_titles.append(title)
    assert len(sf_titles) == 2, f'Unexpected # of supporting fact titles: {len(sf_titles)}'

    num_second_pass_answers = 0
    for paragraph_index, (title, sents) in enumerate(example['context']):
        new_example['paragraphs'][0]['context'] += '\n\n'
        if (num_subanswers > 0) and (title in title2sa_nos.keys()):
            for subanswer_no in (sorted(title2sa_nos[title]) if title2sa_nos is not None else [sf_titles.index(title)]):
                if subanswer_no < num_subanswers:
                    if subqs is not None:
                        new_example['paragraphs'][0]['context'] += subqs[subanswer_no] + ' '
                    new_example['paragraphs'][0]['context'] += ('>' * (subanswer_no + 1)) + ' '

        new_para = ''
        supporting_facts = []
        para_answer_start = None
        for sent_index, sent in enumerate(sents):
            if sent_index == 0:
                sent = sent.lstrip('>')  # NB: Could cause bad answer, but cleans up 2 examples
            if [title, sent_index] in example['supporting_facts']:
                supporting_facts.append(sent)  # Found supporting fact
                if span_q and (title in sf_titles) and (answer in sent) and (para_answer_start is None):  # Find span if possible (Only use 1st span)
                    para_answer_start = len(new_example['paragraphs'][0]['context']) + len(new_para) + sent.index(answer)
            new_para += sent

        if span_q and (title in sf_titles) and (answer in new_para) and (para_answer_start is None):
            para_answer_start = len(new_example['paragraphs'][0]['context']) + new_para.index(answer)
            num_second_pass_answers += 1

        if para_answer_start is not None:
            assert title in sf_titles, f'title {title} not in supporting fact titles: {sf_titles} {example["supporting_facts"]} for new example {new_example["paragraphs"][-1]}'
            answer_starts.append(para_answer_start)
            answer_starts_sf_idxs.append(sf_titles.index(title))
        new_example['paragraphs'][0]['context'] += new_para
        new_example['paragraphs'][0]['supporting_facts'] += supporting_facts

    # Verification
    assert len(answer_starts) > 0, 'Expected example to have >0 answer_starts.'
    for answer_start_sf_idx, answer_start in sorted(zip(answer_starts_sf_idxs, answer_starts)):
        assert answer == new_example['paragraphs'][0]['context'][answer_start: answer_start + len(answer)], f'[{example["_id"]}] answer {answer} != span {new_context[answer_start: answer_start + len(answer)]}'
        new_example['paragraphs'][0]['qas'][0]['answers'].append({"text": answer, "answer_start": answer_start})
    if not use_all_answers:
        new_example['paragraphs'][0]['qas'][0]['answers'] = [new_example['paragraphs'][0]['qas'][0]['answers'][-1]]

    found_sfs = len(new_example['paragraphs'][0]['supporting_facts'])
    missing_sfs = len(example['supporting_facts']) - found_sfs
    return new_example, len(answer_starts), missing_sfs, found_sfs, num_second_pass_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_no', type=int, required=True)  # e.g., 20639223
    parser.add_argument("--num_paragraphs_str", default='1-3', type=str, help="Number of paragraphs used to train single-hop QA model(s)")
    parser.add_argument("--one_to_variable", action='store_true', default=False, help="Assume 1->2 mapping or 1->(variable #) mapping?")
    parser.add_argument("--split", default='dev', type=str, help="Split of data to use")  # e.g., train, dev, hotpot_dev_distractor_adversarial_jiang_v1, hotpot_dev_distractor_adversarial_min_cased_with_sentence_seg_v1
    parser.add_argument('--seed', type=int, default=1, help="Random seed for dataset shuffle (0 for original order)")
    parser.add_argument("--use_subqs", action='store_true', default=False, help="Add SubQs to input?")
    parser.add_argument("--context_is_lower", action='store_true', default=False, help="Whether or not to lowercase the context")
    parser.add_argument("--verbose", action='store_true', default=False, help="Whether or not to print verbosely")
    args = parser.parse_args()
    if args.split == 'hotpot_dev_distractor_adversarial_min_v1':
        print(f'Context is lower-cased for {args.split}.json. Setting args.context_is_lower = True')
        args.context_is_lower = True

    print('Loading supporting facts...')
    one_to_variable_str = 'one_to_variable.' if args.one_to_variable else ''
    task_name = f'hotpot.umt.all.model={args.model_no}.st=0.0.beam=5.lp=1.0.seed=0.atype=sentence-1-center.{one_to_variable_str}subq_model=roberta-large-np={args.num_paragraphs_str}.use_q.use_suba.use_subq.suba1=0.suba2=0'
    supporting_fact_filename = f'{DATA_DIR}/{task_name}/{args.split}.qid2sfs.json'
    sf_sa_nos_filename = supporting_fact_filename.replace('qid2sfs.json', 'qid2sf_sa_nos.json')

    with open(supporting_fact_filename) as f:
        qid2sfs = json.load(f)

    with open(sf_sa_nos_filename) as f:
        qid2sf_sa_nos = json.load(f)

    print('Loading data...')
    with open(f'{DATA_DIR}/hotpot-orig/{args.split}.json') as f:
        data = json.load(f)

    qid2title2sa_nos = {}
    for d in tqdm(data):
        qid2title2sa_nos[d['_id']] = {}
        for (sf_title, sf_sent_no), sa_no in zip(qid2sfs[d['_id']], qid2sf_sa_nos[d['_id']]):
            if sf_title not in qid2title2sa_nos[d['_id']]:
                qid2title2sa_nos[d['_id']][sf_title] = []
            if sa_no not in qid2title2sa_nos[d['_id']][sf_title]:
                qid2title2sa_nos[d['_id']][sf_title].append(sa_no)

    if (args.seed != 0) and (args.split == 'train'):
        random.Random(args.seed).shuffle(data)

    qid2subqs = None
    if args.use_subqs:
        print('Reading in SubQs...')
        with open(f'{DATA_DIR}/hotpot.umt.all.model={args.model_no}.st=0.0.beam=5.lp=1.0.seed=0/{args.split}.json') as f:
            subqs_data = json.load(f)

        qid2subqs = {
            d['paragraphs'][0]['_id']: [qa['question'] for qa in d['paragraphs'][0]['qas']]
            for d in subqs_data['data'] if '_id' in d['paragraphs'][0]
        }

    for num_subas in [1, 2]:
        new_data = {'data': [], 'version': f'{args.split}.json'}
        total_missing_sfs = 0
        total_found_sfs = 0
        num_answer_words = []
        num_question_words = []
        num_answers_per_q = []
        all_num_second_pass_answers = []
        for ex in tqdm(data, desc='Formatting Input'):
            if args.context_is_lower:
                ex['answer'] = ex['answer'].lower()
                for i in range(len(ex['supporting_facts'])):
                    ex['supporting_facts'][i][0] = ex['supporting_facts'][i][0].lower().replace('–', '-').replace('&', 'and').replace('&amp;', 'and')
            new_ex, num_answers_found, ex_missing_sfs, ex_found_sfs, ex_num_second_pass_answers = convert_example(ex, num_subas, args.split != 'train', qid2title2sa_nos[ex['_id']], qid2subqs[ex['_id']] if qid2subqs is not None else None)
            if args.verbose and (num_answers_found > 1):
                print(f'Q:    {new_ex["title"]} ({new_ex["paragraphs"][0]["type"]} / {new_ex["paragraphs"][0]["level"]})')
                print('A:')
                pprint(new_ex['paragraphs'][0]['qas'][0]['answers'])
                print('SFs:')
                pprint(new_ex['paragraphs'][0]['supporting_facts'])
                print('\nCont: ', new_ex['paragraphs'][0]['context'], '\n', '*' * 40)

            new_data['data'].append(new_ex)
            num_answers_per_q.append(num_answers_found)
            all_num_second_pass_answers.append(ex_num_second_pass_answers)
            total_missing_sfs += ex_missing_sfs
            total_found_sfs += ex_found_sfs
            num_answer_words.append(len(ex['answer'].strip().split()))
            num_question_words.append(len(ex['question'].strip().split()))

        num_answers_per_q = np.array(num_answers_per_q)
        print(f'{args.split}: # No Answer Found: {(num_answers_per_q == 0).sum()}')
        print(f'{args.split}: # Answers Found per Example: {num_answers_per_q.mean()}')
        print(f'{args.split}: # Missing SFs: {total_missing_sfs}')
        print(f'{args.split}: # Found SFs: {total_found_sfs}')
        print(f'{args.split}: # 2nd Pass Answers: {sum(all_num_second_pass_answers)}')

        new_data_dir = f'{DATA_DIR}/hotpot.context-long.subqs-{args.model_no}.num_subas-{num_subas}'
        if args.seed != 0:
            new_data_dir += f'.shuffled-{args.seed}'
        if args.use_subqs:
            new_data_dir += f'.use_subqs'
        os.makedirs(new_data_dir, exist_ok=True)
        save_filename = os.path.join(new_data_dir, f'{args.split}.json')
        with open(save_filename, 'w') as f:
            json.dump(new_data, f, indent=2, sort_keys=False)
        print('Saved to:', save_filename)
