import argparse
import fasttext
import numpy as np
import os
import pandas as pd
import random
import shutil
from tqdm.auto import tqdm
from transformers.data.processors.glue import glue_processors
from transformers.loss_data_utils import tune_std, tune_temperature


train_parameters = {
    'lr': 0.1,
    'dim': 300,
    'ws': 5,
    'epoch': 5,
    'minCount': 1,
    'minCountLabel': 0,
    'minn': 0,
    'maxn': 0,
    'neg': 5,
    'wordNgrams': 1,
    'bucket': 2000000,
    'thread': 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': '__label__',
    'verbose': 5,
    'pretrainedVectors': '',
    'seed': 0,
}


def get_model_parameters(model):
    args_getter = model.f.getArgs()

    parameters = {}
    for param in train_parameters:
        attr = getattr(args_getter, param)
        if param == 'loss':
            attr = attr.name
        parameters[param] = attr

    return parameters


def evaluate(eval_model, dataset, split, labels, output_dir, suffix='', regression=False, format_kwargs={}):
    gold_answer_probs = []
    accs = []
    preds_csv = {f'p({label_no})': [] for label_no in range(len(labels) if not regression else 1)}
    preds_csv['label'] = []
    for d in tqdm(dataset[split], desc=f'Evaluating on {split}'):
        pred_labels, pred_probs = eval_model.predict(format_example(d, **format_kwargs), len(labels))
        assert np.all(pred_probs == np.array(sorted(pred_probs, reverse=True)))
        label2prob = {label[len('__label__'):]: prob for label, prob in zip(pred_labels, pred_probs.tolist())}
        if not regression:
            gold_answer_probs.append(label2prob[d.label])
            accs.append(pred_labels[0][len('__label__'):] == d.label)
            for label, prob in label2prob.items():
                preds_csv[f'p({labels.index(label)})'].append(prob)
            preds_csv['label'].append(labels.index(d.label))
        else:
            pred = sum(prob * float(label) for label, prob in label2prob.items())
            preds_csv[f'p(0)'].append(pred)
            preds_csv['label'].append(float(d.label))

    save_split = "val" if split == "dev" else split
    df = pd.DataFrame(preds_csv)
    df.to_csv(f'{output_dir}/preds.{save_split}.csv', index=False)

    stats = {
        f'{save_split}_loss{suffix}': -np.log(np.array(gold_answer_probs)).mean(),
        f'{save_split}_acc{suffix}': np.mean(accs),
    } if not regression else {
        f'{save_split}_loss{suffix}': ((df['p(0)'] - df['label']) ** 2).mean(),
    }
    return stats, df


def format_example(ex, sent1_upper=False, sent2_upper=False, replace_newline=False):
    sent1 = ex.text_a.strip()
    if sent1_upper:
        sent1 = sent1.upper()
    if ex.text_b is None:
        return sent1
    sent2 = ex.text_b.strip()
    if sent2_upper:
        sent2 = sent2.upper()
    formatted = f'{sent1} {sent2}'
    if replace_newline:
        formatted = formatted.replace('\n', '\t')
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="The name of the folder containing train/dev/test data.")
    parser.add_argument("--model_name", type=str, default='fasttext', help="The name of the fasttext model to use.",
                        choices=['fasttext', 'fasttext_no_pretrain', 'fasttext_sent1_upper', 'fasttext_sent2_upper',
                                 'fasttext_no_pretrain_sent1_upper', 'fasttext_no_pretrain_sent2_upper'])
    parser.add_argument("--autotune_duration", type=int, default=0,
                        help="Number of seconds to use for autotuning hyperparameters. No autotune by default.")
    parser.add_argument("--seeds", type=str, default=[12, 20, 21, 22, 23], nargs='+', help="Seeds to load, e.g., '12 20 21 22 23' (if not using default).")
    parser.add_argument("--prequential_block_no", required=True, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train). "
                        "Use -1 for --preprocess_only and --prequential_num_blocks value to train on all prequential code examples and evaluate on test.")
    parser.add_argument("--prequential_num_blocks", default=8, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train).")
    parser.add_argument("--prequential_min_block_size", default=64, type=int, help="The minimum number of examples to use in a training block (inclusive).")
    parser.add_argument("--prequential_max_block_size", default=10000, type=int, help="The maximum number of examples to use in a training block (exclusive). -1 for all examples.")
    parser.add_argument("--prequential_valid_frac", default=0.1, type=float, help="The fraction of examples to use for validation.")
    parser.add_argument("--train_seed", default=None, type=int, help="The seed to use for randomness during training (not for data sampling), if different from default seed.")
    parser.add_argument("--thread", type=int, default=1, help="Number of threads to use (shouldn't exceed number of CPUs used.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assert args.prequential_block_no <= args.prequential_num_blocks, f'Expected --prequential_block_no ({args.prequential_block_no}) <= --prequential_num_blocks ({args.prequential_num_blocks})'
    assert args.prequential_block_no >= 0, f'Expected --prequential_block_no ({args.prequential_block_no}) >= 0'
    if all([os.path.exists(f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{args.task_name}.mn-{args.model_name}.ad-{args.autotune_duration}.seed-{seed}.pbn-{args.prequential_block_no}/test_results.txt') for seed in args.seeds]):
        print('Done training all seeds!! (Already finished)')
        return

    best_params = train_fasttext_classifier(args, args.seeds[0])
    for seed in args.seeds[1:]:
        train_fasttext_classifier(args, seed, best_params)
    print('Done training all seeds!!')


def train_fasttext_classifier(args, seed, params=None):
    task_dir = f'{os.environ["BASE_DIR"]}/data/{args.task_name}'
    output_dir = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{args.task_name}.mn-{args.model_name}.ad-{args.autotune_duration}.seed-{seed}.pbn-{args.prequential_block_no}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    regression = args.task_name.lower().startswith('sts-b')
    round_to_nearest = 0.2 if regression else None

    split_file_format = task_dir + '/{}' + f'.mn-{args.model_name}.pbn-{args.prequential_block_no}.seed-{seed}.txt'
    data = {}
    data_lines = {}
    processor_name = args.task_name.split('.', 1)[0]
    processor = glue_processors[processor_name]()
    format_kwargs = {
        'sent1_upper': ("sent1_upper" in args.model_name),
        'sent2_upper': ("sent2_upper" in args.model_name),
        'replace_newline': args.task_name.lower().startswith('anli'),
    }
    for split in ['train', 'dev', 'test']:
        print(f'Loading {args.task_name} {split} data...')
        if args.prequential_block_no < args.prequential_num_blocks:
            data[split] = getattr(processor, f'get_train_examples')(task_dir)
            prequential_max_num_train_samples = len(data[split]) if args.prequential_max_block_size == -1 else min(args.prequential_max_block_size, len(data[split]))
            random.Random(seed).shuffle(data[split])
            block_start_idxs = np.round(np.logspace(np.log10(args.prequential_min_block_size), np.log10(prequential_max_num_train_samples), args.prequential_num_blocks + 1)).astype(int)
            num_train_samples = int(round((1. - args.prequential_valid_frac) * block_start_idxs[args.prequential_block_no]))
            mode2slice = {
                'train': slice(None, num_train_samples),
                'dev': slice(num_train_samples, block_start_idxs[args.prequential_block_no]),
                'test': slice(block_start_idxs[args.prequential_block_no], block_start_idxs[args.prequential_block_no + 1]),
            }
            assert split in {'train', 'dev', 'test'}, f'Unexpected mode = {split} not in ["train", "dev", "test"]'
            data[split] = data[split][mode2slice[split]]
        else:
            data[split] = getattr(processor, f'get_{split}_examples')(task_dir)
            if split == 'train':
                prequential_max_num_train_samples = len(data[split]) if args.prequential_max_block_size == -1 else min(args.prequential_max_block_size, len(data[split]))
                random.Random(seed).shuffle(data[split])
                data[split] = data[split][:prequential_max_num_train_samples]

        print(f'Saving {len(data[split])} {args.task_name} {split} examples...')
        data_lines[split] = [f'__label__{(round(float(d.label) / round_to_nearest) * round_to_nearest) if regression else d.label} {format_example(d, **format_kwargs)}'
                             for d in tqdm(data[split])]
        with open(split_file_format.format(split), 'w') as f:
            f.writelines('\n'.join(data_lines[split]) + '\n')

    print(f'Training seed {seed}...')
    train_seed = args.train_seed if args.train_seed is not None else seed
    if params is None:
        params = {
            'input': split_file_format.format('train'),
            'verbose': 5,
            'thread': args.thread,
            'autotuneValidationFile': split_file_format.format('dev'),
            'autotuneDuration': args.autotune_duration,
            'seed': train_seed,
            # 'autotuneMetric': 'f1',
        }
        if 'no_pretrain' not in args.model_name:
            params['pretrainedVectors'] = f'{os.environ["BASE_DIR"]}/data/fastText/crawl-300d-2M.vec'
            params['dim'] = 300
        if args.autotune_duration == 0:
            for k in ['autotuneValidationFile', 'autotuneDuration', 'autotuneMetric']:
                params.pop(k, None)
    else:
        params['input'] = split_file_format.format('train')
        params['seed'] = train_seed

    model = fasttext.train_supervised(**params)

    df = {}
    stats = {}
    for split in ['dev', 'test']:
        label_list = np.arange(0, 5 + round_to_nearest, round_to_nearest).tolist() if regression else processor.get_labels()  # NB: STS-B specific range
        split_results, df[split] = evaluate(model, data, split, label_list, output_dir, suffix='_no_temperature', regression=regression, format_kwargs=format_kwargs)
        stats.update(split_results)

    print('Tuning temperature on the dev set...')
    stats.update((tune_std if regression else tune_temperature)(df, verbose=True))

    print('Saving calibrated results...')
    with open(f'{output_dir}/test_results.txt', 'w') as f:
        for k, v in stats.items():
            f.writelines(f'{k} = {v}\n')
    for k, v in stats.items():
        print(f'{k} = {v}')

    print(f'Done training seed {seed}! Saved results to: {output_dir}/test_results.txt')
    
    best_params = get_model_parameters(model)
    del model
    return best_params


if __name__ == '__main__':
    main()
