import argparse
from collections import Counter
from copy import deepcopy
import os
import re
import string
from tqdm import tqdm
from difflib import SequenceMatcher
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = os.path.join(os.environ['MAIN_DIR'], 'XLM/data')


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_metrics(prediction, gold):
    """
    Compute a vector of Exact Match scores (one element per question)
    """
    ems = []
    qid2metric2value = {}
    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
        else:
            em = float(exact_match_score(prediction['answer'][cur_id], dp['answer']))
            f1, prec, recall = f1_score(prediction['answer'][cur_id], dp['answer'])
            ems.append(em)
            # NB: Can add other metrics here as needed. See examples/hotpot_evaluate_v1.py for more metrics
            qid2metric2value[cur_id] = {'em': em, 'f1': f1, 'prec': prec, 'recall': recall, 'gold': dp['answer'], 'predicted': prediction['answer'][cur_id]}
    return np.array(ems), qid2metric2value


def get_task_name_from_path(path):
    if '.mn=' in path:
        return path.split('.tn=')[1].split('.mn=')[0]
    elif '/mn=' in path:
        return path.split('.tn=')[1].split('/mn=')[0]
    else:
        raise Exception()


def parse_preds_with_metadata(preds_files, gold_file, split,  task_name=None):

    file2preds = {file: json.load(open(file)) for file in preds_files + [gold_file]}
    qid2example = {example['_id']: example for example in file2preds[gold_file]}
    file2qid2metrics = {file: compute_metrics(file2preds[file], file2preds[gold_file])[1] for file in preds_files}

    file2qid2info = {}
    for file in preds_files:
        qid2info = deepcopy(qid2example)
        for qid, pred in file2preds[file]['answer'].items():
            qid2info[qid]['pred'] = pred

        if task_name is None:
            task_name = get_task_name_from_path(file)
        try:
            with open(os.path.join(DATA_DIR, task_name, f'{split}.json')) as f:
                data = json.load(f)
        except:
            with open(os.path.join(DATA_DIR, task_name.replace('-squad.medium_hard_frac=1.0', ''), f'{split}.json')) as f:
                data = json.load(f)

        for article in data['data']:
            for paragraph in article['paragraphs']:
                if '_id' not in paragraph:
                    continue  # Skip: It's a SQuAD question
                qid = paragraph['_id']
                if 'subqs' in qid2info[qid]:
                    continue  # Already added the subqs/subas for this example
                q_subqas = paragraph['qas'][0]['question'].strip()
                q_split = q_subqas.split('//')
                subqas = q_split[1:]
                qid2info[qid]['subqas'] = '//'.join(subqas)
                qid2info[qid]['subqs'] = []
                qid2info[qid]['subas'] = []
                qid2info[qid]['probability_subas'] = paragraph['qas'][0].get('probability_subas', [])
                if len(subqas) > 0:  # Only add subqs/subas if they exist in input
                    for subqa in subqas:
                        subqa = subqa.strip()
                        subqa_split = subqa.split('/')
                        subq = subqa_split[0].strip()
                        suba = subqa_split[1].strip() if len(subqa_split) > 1 else ''
                        qid2info[qid]['subqs'].append(subq)
                        qid2info[qid]['subas'].append(suba)
                for metric, value in file2qid2metrics[file][qid].items():
                    assert metric not in qid2info[
                        qid], f'Metric name {metric} already exists in example info. Please change the metric name. {qid2info[qid]}'
                    qid2info[qid][metric] = value
        file2qid2info[file] = qid2info

    return file2preds, file2qid2metrics, file2qid2info


def lcs_fast(a, b):
    s = SequenceMatcher(None, a, b)
    return s.find_longest_match(0, len(a), 0, len(b)).size


def simple_text(text):
    return re.compile('[\W_]+').sub('', text.strip().lower())


def get_sps_from_sub_answers(info, strategy='biggest', thresh=0.8, min_sent_len=10, orig_info=None):
    assert strategy in ['biggest', 'all']
    contexts = info['context']
    if orig_info is not None:
        contexts = orig_info['context']

    sub_as = info['subas']
    sps = []
    for title, sents in contexts:
        ssents = [simple_text(sent) for sent in sents]
        sssents = [set(ssent) for ssent in ssents]
        for sub_a in sub_as:
            ssub_a = simple_text(sub_a)
            match_fractions = []
            for sent, ssent, sssent in zip(sents, ssents, sssents):
                if len(ssent) < min_sent_len:
                    mf = 0
                else:
                    if (len(sssent.intersection(ssub_a)) / len(sssent)) > 0.7:
                        if ssent == ssub_a:
                            mf = 1.
                        elif ssent in ssub_a:
                            mf = 1.
                        else:
                            mf = lcs_fast(ssent, ssub_a) / len(ssent)
                    else:
                        mf = 0
                match_fractions.append(mf if mf > thresh else 0.)
            if all([mf == 0 for mf in match_fractions]):
                continue
            if strategy == 'all':
                for i, mf in enumerate(match_fractions):
                    if mf > 0.:
                        sps.append([title, i])
            elif strategy == 'biggest':
                sps.append([title, np.argmax(match_fractions)])
    return sps


def get_sps_from_question_with_bow(info, topk=5):
    q = info['question']
    qbow = set(q.lower().strip().split())#.difference(stp)
    contexts = info['context']

    sp_col = []

    for title, sents in contexts:
        for j, s in enumerate(sents):
            sbow = set(s.lower().strip().split())#.difference(stp)
            score = len(qbow.intersection(sbow)) / len(qbow.union(sbow))
            sp_col.append(([title, j], score))
    best_sps, best_scores = zip(*sorted(sp_col, key=lambda x: -x[1]))
    return best_sps[:topk]


def get_sps_from_question_with_tfidf(info, tf, topk=5):
    q = info['question']
    contexts = info['context']
    search_space = []
    sp_labels = []
    for title, sents in contexts:
        for j, s in enumerate(sents):
            search_space.append(s)
            sp_labels.append([title, j])

    query = tf.transform([q])
    ss = tf.transform(search_space)
    scores = linear_kernel(query, ss)[0]
    aranked = np.argsort(-scores)
    to_return = [[a[0], int(a[1])] for a in list(np.array(sp_labels)[aranked])[:topk]]
    return to_return


def build_tfidf(infos):
    corpus = []
    for info in infos:
        for title, sents in info['context']:
            for j, s in enumerate(sents):
                corpus.append(s)
        corpus.append(info['question'])
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                         min_df=0, sublinear_tf=True, norm='l2')
    tf.fit_transform(corpus)
    return tf


def get_sps_from_final_answer(info):
    sps = []
    contexts = info['context']
    ans = info['predicted']
    sans = simple_text(ans)

    for title, sents in contexts:
        ssents = [simple_text(sent) for sent in sents]
        sssents = [set(ssent) for ssent in ssents]
        for i, (sent, ssent, sssent) in enumerate(zip(sents, ssents, sssents)):
            if sans in ssent:
                sps.append([title, i])
    return sps


def get_sps(info, use_sub_answers, use_final_answers, strategy='all', thresh=0.8, min_sent_len=10, orig_info=None):
    sps = []
    if use_sub_answers:
        sub_answer_sps = get_sps_from_sub_answers(info,  strategy=strategy, thresh=thresh, min_sent_len=min_sent_len, orig_info=orig_info)
        sps += sub_answer_sps
    if use_final_answers:
        final_sps = get_sps_from_final_answer(info)
        sps += final_sps
    return sps


def get_final_answer_gold_sps(dp):
    a = dp['answer']
    gold_sups = []

    if a in ['yes', 'no']:
        gold_sups = []

    counts = 0
    ev_counts = 0
    sups = []
    for t, ind in dp['supporting_facts']:
        for ti, cs in dp['context']:
            if ti.lower() == t.lower():
                for indi, c in enumerate(cs):
                    if indi == ind:
                        sups.append((t, indi, c))

    for t, indi, c in sups:
        counts += c.lower().count(a.lower())
        ev_counts += (c.lower().count(a.lower()) != 0)
        if c.lower().count(a.lower()) != 0:
            gold_sups.append((t, indi))
    return gold_sups


def get_sp_score(prediction, gold, final_sps):
    cur_sp_pred = {(s[0].lower(), s[1]) for s in prediction}
    gold_sp_pred = {(s[0].lower(), s[1]) for s in gold}
    final_sps = {(s[0].lower(), s[1]) for s in final_sps}

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    hits = len(cur_sp_pred.intersection(gold_sp_pred)) > 0
    two_hits = len(cur_sp_pred.intersection(gold_sp_pred)) > 1

    non_final_gold_sps = gold_sp_pred.difference(final_sps)
    if len(non_final_gold_sps) == 0:
        non_final_hits = None
    else:
        non_final_hits = len(cur_sp_pred.intersection(non_final_gold_sps)) > 0
    if len(non_final_gold_sps) < 2:
        non_final_two_hits = None
    else:
        non_final_two_hits = len(cur_sp_pred.intersection(non_final_gold_sps)) > 1

    return em, f1, prec, recall, hits, two_hits, non_final_hits, non_final_two_hits


def get_sp_score_paragraph_level(prediction, gold, final_sps):
    cur_sp_pred = set([p[0].lower() for p in prediction])
    gold_sp_pred = set([g[0].lower() for g in gold])
    final_sps = set([g[0].lower() for g in final_sps])
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    hits = len(cur_sp_pred.intersection(gold_sp_pred)) != 0
    if len(gold_sp_pred) == 1:
        two_hits = None
    else:
        two_hits = len(cur_sp_pred.intersection(gold_sp_pred)) > 1

    non_final_gold_sps = gold_sp_pred.difference(final_sps)
    if len(non_final_gold_sps) == 0:
        non_final_hits = None
    else:
        non_final_hits = len(cur_sp_pred.intersection(non_final_gold_sps)) > 0
    if len(non_final_gold_sps) < 2:
        non_final_two_hits = None
    else:
        non_final_two_hits = len(cur_sp_pred.intersection(non_final_gold_sps)) > 1

    return em, f1, prec, recall, hits, two_hits, non_final_hits, non_final_two_hits


def main(pred_files, gold_file, split, task_name, use_sub_answers, use_final_answers, fs_for_saving_name, do_save=True):
    """
    Calculate supporting fact retrieval analysis

    Interesting cases:
    * SP when using sub answer sentences only
    * SP when using final answer sentence
    * SP when using sub answer setnences and final answer sentence

    in case of sub answers that span

    Both at sentence level (official) and paragraph level

    :param pred_files: files to evaluate
    :param gold_file: hotpot orig
    :return: ?
    """
    do_sps = True
    do_sps_para = True
    do_bins = True
    file2preds, file2qid2metrics, file2qid2info = parse_preds_with_metadata(pred_files, gold_file, split, task_name=task_name)

    with open(f'{DATA_DIR}/data/hotpot-orig/dev2.qids.txt') as f:
        ids_to_do = [l for l in f.read().split('\n') if l != '']
    ids_to_do = set(ids_to_do)

    orig_gold_data = {}
    with open(f'{DATA_DIR}/hotpot-orig/dev.json') as f:
        original_ds = json.load(f)
        for d in original_ds:
            orig_gold_data[d['_id']] = d

    gold_data = {}
    with open(gold_file) as f:
        ds = json.load(f)
        for d in ds:
            gold_data[d['_id']] = d

    metric_names = ['all_ems',
        'all_f1s',
        'all_precs',
        'all_recalls',
        'all_hits',
        'all_two_hits',
        'all_nonfinal_hits',
        'all_nonfinal_two_hits'
    ]
    all_metrics = {m: [] for m in metric_names}
    all_p_metrics = {m: [] for m in metric_names}
    tf = build_tfidf(file2qid2info[pred_files[0]].values())

    low_score_acc = []
    high_score_acc = []
    n_bins = 5
    binned_f1s = {k:[] for k in range(n_bins)}
    all_fs_for_saving = {}

    has_zeros = []
    all_zeros = []
    perc_zeros = []
    no_facts = []
    first_hits = []

    for file in pred_files:
        metrics = {m: [] for m in metric_names}
        p_metrics = {m: [] for m in metric_names}
        cou = 0

        ems_for_binning = []
        fs1_for_binning = []

        for qid, info in tqdm(file2qid2info[file].items()):

            if qid not in ids_to_do:
                continue

            cou += 1
            gold_sps = info['supporting_facts']
            sps = get_sps(info, use_sub_answers, use_final_answers, orig_info=orig_gold_data[qid] if 'min' in split else None)
            has_zeros.append(any([s[1] == 0 for s in sps]))
            all_zeros.append(all([s[1] == 0 for s in sps]))
            perc_zeros.append(sum([s[1] == 0 for s in sps])/ len(sps) if len(sps) > 0 else 0.)
            no_facts.append(len(sps) == 0)
            gold_sps_first_sent = {g[0] for g in gold_sps if g[1] == 0}
            sp_first_sent = {g[0] for g in sps if g[1] == 0}
            first_hits.append(len(gold_sps_first_sent.intersection(sp_first_sent)) > 0)

            # bow_sps = get_sps_from_question_with_bow(info, topk=2)
            # tfidf_sps = get_sps_from_question_with_tfidf(info, tf, topk=3)
            # sps = tfidf_sps

            final_gold_sps = get_final_answer_gold_sps(gold_data[qid])
            mets = get_sp_score(sps, gold_sps, final_gold_sps)
            p_mets = get_sp_score_paragraph_level(sps, gold_sps, final_gold_sps)
            for n, m in zip(metric_names, mets):
                metrics[n].append(m)
            for n, m in zip(metric_names, p_mets):
                p_metrics[n].append(m)

            ems_for_binning.append((mets[0], qid))
            fs1_for_binning.append((mets[1], qid))

        sorted_scores, sorted_qids = zip(*sorted(ems_for_binning, key=lambda x:x[0]))
        bin_split = sorted_scores.index(1.)
        low_score_qids, high_score_qids = sorted_qids[:bin_split], sorted_qids[bin_split:]
        low_scores = np.mean([file2qid2metrics[file][qid]['f1'] for qid in low_score_qids])
        high_scores = np.mean([file2qid2metrics[file][qid]['f1'] for qid in high_score_qids])
        low_score_acc.append(low_scores)
        high_score_acc.append(high_scores)
        fs_for_saving = {qid: {'ans_f1': file2qid2metrics[file][qid]['f1'], 'sp_f1': f1} for f1, qid in fs1_for_binning}
        all_fs_for_saving[file] = fs_for_saving

        sorted_scores, sorted_qids = zip(*sorted(fs1_for_binning, key=lambda x:x[0]))
        bin_size = len(sorted_qids) // n_bins
        for b in range(n_bins):
            binned_ids = sorted_qids[b * bin_size: (b+1) * bin_size]
            binned_f1s[b].append(np.mean([file2qid2metrics[file][qid]['f1'] for qid in binned_ids]))

        for m in metric_names:
            all_metrics[m].append(np.mean([v for v in metrics[m] if v is not None]))
        for m in metric_names:
            all_p_metrics[m].append(np.mean([v for v in p_metrics[m] if v is not None]))

    if do_bins:
        print('Binned Results:')
        print(f'em=0 score f1s: {np.mean(low_score_acc)} +/- {np.std(low_score_acc)}')
        print(f'em=1 score f1s: {np.mean(high_score_acc)} +/- {np.std(high_score_acc)}')
        print('F1 binned results:')
        for n in range(n_bins):
            print(f'f1 bin {n} score f1s: {np.mean(binned_f1s[n])} +/- {np.std(binned_f1s[n])}')
        if do_save:
            with open(fs_for_saving_name, 'w') as f:
                json.dump(all_fs_for_saving, f)

    if do_sps:
        print('First Sentence retrieval analysis:')
        print(f'Has Zeros: {np.mean(has_zeros)}')
        print(f'All zeros: {np.mean(all_zeros)}')
        print(f'perc zeros: {np.mean(perc_zeros)}')
        print(f'No Facts: {np.mean(no_facts)}')
        print(f'first hits: {np.mean(first_hits)}')
        print('Evidence (sentence level):')
        for m in metric_names:
            print(f'{m}:', np.mean(all_metrics[m]), '+/-', np.std(all_metrics[m]))
        sentence_means = [np.mean(all_metrics[m]) for m in metric_names] + [np.mean(has_zeros), np.mean(all_zeros), np.mean(perc_zeros), np.mean(no_facts), np.mean(first_hits)]
        sentence_stds = [np.std(all_metrics[m]) for m in metric_names] + [0., 0., 0., 0., 0.]
        assert len(sentence_means) == len(sentence_stds)
        print(','.join([
            str(v) for p in zip(sentence_means, sentence_stds) for v in p
        ]))

    if do_sps_para:
        print('Evidence (paragraph level):')
        for m in metric_names:
            print(f'{m}:', np.mean(all_p_metrics[m]), '+/-', np.std(all_p_metrics[m]))
        paragraph_means = [np.mean(all_p_metrics[m]) for m in metric_names]
        paragraph_stds = [np.std(all_p_metrics[m]) for m in metric_names]
        print(','.join([
            str(v) for p in zip(paragraph_means, paragraph_stds) for v in p
        ]))

    return sentence_means, sentence_stds, paragraph_means, paragraph_stds


def analyze_supporting_fact_retrieval():
    seeds = 5
    path_to_model_to_test = ''
    task_name = get_task_name_from_path(path_to_model_to_test)
    use_sub_answers = True
    use_final_answers = False

    eval_sets = {
        'dev': 'hotpot_predictions_gn_dev.json',
        'adv': 'hotpot_predictions_gn_hotpot_dev_distractor_adversarial_jiang_v1.json',
        'min': 'hotpot_predictions_gn_hotpot_dev_distractor_adversarial_min_v1.json'
    }
    gold_files = {
        'dev':  'dev',
        'adv':  'hotpot_dev_distractor_adversarial_jiang_v1',
        'min':  'hotpot_dev_distractor_adversarial_min_v1'
    }
    fs_for_saving_name = './sp_f1_and_f1_final_model_{}_{}_{}.json'

    for split in ['dev', 'adv', 'min']:
        gold_file = os.path.join(DATA_DIR, f'hotpot-orig/{gold_files[split]}.json')
        paths_to_model_to_test = [os.path.join(path_to_model_to_test.format(s), eval_sets[split]) for s in range(seeds)]

        main(
            paths_to_model_to_test,
            gold_file,
            gold_files[split],
            task_name,
            use_sub_answers,
            use_final_answers,
            fs_for_saving_name.format(split, use_sub_answers, use_final_answers)
        )


def parse_preds_with_metadata_simple(file, gold_file, split, task_name=None):

    with open(gold_file) as f:
        qid2info = {example['_id']: example for example in json.load(f)}

    if task_name is None:
        task_name = get_task_name_from_path(file)
    try:
        with open(os.path.join(DATA_DIR, task_name, f'{split}.json')) as f:
            data = json.load(f)
    except:
        with open(os.path.join(DATA_DIR, task_name.replace('-squad.medium_hard_frac=1.0', ''), f'{split}.json')) as f:
            data = json.load(f)

    for article in data['data']:
        for paragraph in article['paragraphs']:
            if '_id' not in paragraph:
                continue  # Skip: It's a SQuAD question
            qid = paragraph['_id']
            if 'subqs' in qid2info[qid]:
                continue  # Already added the subqs/subas for this example
            q_subqas = paragraph['qas'][0]['question'].strip()
            q_split = q_subqas.split('//')
            subqas = q_split[1:]
            qid2info[qid]['subqas'] = '//'.join(subqas)
            qid2info[qid]['subqs'] = []
            qid2info[qid]['subas'] = []
            qid2info[qid]['probability_subas'] = paragraph['qas'][0].get('probability_subas', [])
            if len(subqas) > 0:  # Only add subqs/subas if they exist in input
                for subqa in subqas:
                    subqa = subqa.strip()
                    subqa_split = subqa.split('/')
                    subq = subqa_split[0].strip()
                    suba = subqa_split[1].strip() if len(subqa_split) > 1 else ''
                    qid2info[qid]['subqs'].append(subq)
                    qid2info[qid]['subas'].append(suba)

    return qid2info


def eval_supporting_fact_retrieval(pred_file, gold_file, split, task_name, dedup_facts=True):
    """
    Calculate supporting fact retrieval analysis

    Interesting cases:
    * SP when using sub answer sentences only
    * SP when using final answer sentence
    * SP when using sub answer setnences and final answer sentence

    in case of sub answers that span

    Both at sentence level (official) and paragraph level

    :param pred_files: files to evaluate
    :param gold_file: hotpot orig
    :return: ?
    """
    print('Evaluating Support Fact Scores!!')
    if dedup_facts:
        print('Deduplicating repeated supporting facts')
    use_final_answers = False
    use_sub_answers=True

    qid2info = parse_preds_with_metadata_simple(pred_file, gold_file, split, task_name=task_name)

    # only needed for Min data I think
    orig_gold_data = {}
    with open(f'{DATA_DIR}/hotpot-orig/dev.json') as f:
        original_ds = json.load(f)
        for d in original_ds:
            orig_gold_data[d['_id']] = d

    gold_data = {}
    with open(gold_file) as f:
        ds = json.load(f)
        for d in ds:
            gold_data[d['_id']] = d

    metric_names = [
        'all_ems',
        'all_f1s',
        'all_precs',
        'all_recalls',
        'all_hits',
        'all_two_hits',
        'all_nonfinal_hits',
        'all_nonfinal_two_hits'
    ]

    metrics = {m: [] for m in metric_names}
    p_metrics = {m: [] for m in metric_names}

    qid2sps = {}
    for qid, info in tqdm(qid2info.items(), desc='Evaluating Supporting Facts'):
        gold_sps = info['supporting_facts']
        sps = get_sps(info, use_sub_answers, use_final_answers, orig_info=orig_gold_data[qid] if 'min' in split else None)

        if dedup_facts:
            unique_sps = []
            for sp in sps:
                sp = tuple(sp)
                if sp not in unique_sps:
                    unique_sps.append(sp)
            sps = [list(sp) for sp in unique_sps]
        qid2sps[qid] = sps

        final_gold_sps = get_final_answer_gold_sps(gold_data[qid])
        mets = get_sp_score(sps, gold_sps, final_gold_sps)
        p_mets = get_sp_score_paragraph_level(sps, gold_sps, final_gold_sps)
        for n, m in zip(metric_names, mets):
            metrics[n].append(m)
        for n, m in zip(metric_names, p_mets):
            p_metrics[n].append(m)

    print('\nSupporting fact metrics:')
    for m in metrics:
        print('\t', m, round(100. * np.mean([v for v in metrics[m] if v is not None]), 3))

    print('\nSupporting fact paragraph metrics:')
    for m in metric_names:
        print('\t', m, round(100. * np.mean([v for v in p_metrics[m] if v is not None]), 3))

    return qid2sps


if __name__ == '__main__':
    # analyze_supporting_fact_retrieval()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_no', type=int, required=True)
    parser.add_argument("--num_paragraphs_str", default='1-3', type=str, help="Number of paragraphs used to train single-hop QA model(s)")
    parser.add_argument("--one_to_variable", action='store_true', default=False, help="Assume 1->2 mapping or 1->(variable #) mapping?")
    parser.add_argument("--split", default='dev', type=str, help="Split of data to use")
    args = parser.parse_args()

    one_to_variable_str = 'one_to_variable.' if args.one_to_variable else ''
    task_name = f'hotpot.umt.all.model={args.model_no}.st=0.0.beam=5.lp=1.0.seed=0.atype=sentence-1-center.{one_to_variable_str}subq_model=roberta-large-np={args.num_paragraphs_str}.use_q.use_suba.use_subq.suba1=0.suba2=0'
    eval_supporting_fact_retrieval(f'{DATA_DIR}/{task_name}', f'{DATA_DIR}/hotpot-orig/{args.split}.json', args.split, task_name)
