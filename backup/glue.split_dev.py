import os
import random

seed = 42
for task_name in ["cola", "mnli", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli"]:
    # Setup
    print(f'Splitting {task_name} dev...')
    data_dir = f'{os.environ["BASE_DIR"]}/data/{task_name}'
    task = task_name.split('.', 1)[0]
    data_start_row = 0 if ((task == 'cola') or ('anli' in task)) else 1
    split_postfixes = ['_matched', '_mismatched'] if task == 'mnli' else ['']

    for split_postfix in split_postfixes:
        # Rename existing dev/test files (keep backups)
        for split in ['dev', 'test']:
            old_filename = f'{data_dir}/{split}{split_postfix}.tsv'
            new_filename = f'{data_dir}/{split}{split_postfix}_orig.tsv'
            assert os.path.exists(old_filename)
            assert not os.path.exists(new_filename)
            os.rename(old_filename, new_filename)

        # Load data and header (if applicable)
        with open(f'{data_dir}/dev{split_postfix}_orig.tsv') as f:
            lines = [line for line in f]
        header = lines[:data_start_row]
        data = lines[data_start_row:]

        # Randomly split dev data
        rand = random.Random(seed)
        rand.shuffle(data)
        data_mid_row = len(data) // 2

        # Save dev splits
        with open(f'{os.environ["BASE_DIR"]}/data/{task_name}/dev{split_postfix}.tsv', 'w') as f:
            for line in (header + data[:data_mid_row]):
                f.write(line)

        with open(f'{os.environ["BASE_DIR"]}/data/{task_name}/test{split_postfix}.tsv', 'w') as f:
            for line in (header + data[data_mid_row:]):
                f.write(line)

print('Done!')
