import argparse
import json
import os
import random
from transformers.data.processors.glue import glue_processors


DATA_DIR = f'{os.environ["BASE_DIR"]}/data'
VERSION = 2
shuffle_rng = random.Random(42)
replace_fraction_rng = random.Random(42)
simple_save_dir_tasks = {"cola", "mrpc", "sts-b", "rte", "wnli", "qnli", "qqp", "sst-2", "anli.round1", "anli.round2", "anli.round3"}


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
    print(f'Loaded {len(data)} {args.task} examples')

    transform2data = {
        'original': [
            {
                'guid': d.guid,
                'text_a': d.text_a,
                'text_b': d.text_b if two_input_task else None,
                'label': d.label,
            } for d in data
        ]
    }

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

