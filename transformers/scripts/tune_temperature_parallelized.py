import argparse
from multiprocessing import Pool, cpu_count
import os
from tqdm.auto import tqdm
from transformers.loss_data_utils import mn2hpstrs, tn2max_num_samples, group2datasethps, save_temperature_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="", type=str, required=True, help="The group of tasks to run on, e.g. esnli",
                        choices=list(sorted(group2datasethps.keys())))
    parser.add_argument("--prequential_num_blocks", default=8, type=int,
                        help="The number of blocks sent when calculating prequential code length (number of models to train).")
    args = parser.parse_args()

    tns = group2datasethps[args.group]['tn']
    max_num_samples = tn2max_num_samples(tns[0])
    for tn in tns:
        assert max_num_samples == tn2max_num_samples(tn), f'Expected tn2max_num_samples("{tn}") == {max_num_samples} but got {tn2max_num_samples(tn)}'

    skip_mns = {'fasttext'}
    mns = [mn for mn in group2datasethps[args.group]['mn'] if mn not in skip_mns]
    pbns = list(range(args.prequential_num_blocks + 1))
    seeds = group2datasethps[args.group]['seed']

    save_dirs = []
    for mn in mns:
        if 'fasttext' in mn:
            continue
        for tn in tns:
            cached_test_results_file = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{tn}.mn-{mn}.pbn2seed2hpstr2stats.json'
            if os.path.exists(cached_test_results_file):
                print('Skipping! Found cached_test_results_file:', cached_test_results_file)
                continue
            for pbn in pbns:
                for seed in seeds:
                    hpstrs = mn2hpstrs(mn)
                    if (seed != seeds[0]) and ('bs-' in hpstrs[0]) and ('lr-' in hpstrs[0]) and ('nte-' in hpstrs[0]):
                        hpstrs = [f'bs-0.lr-0.nte-{hpstrs[0].split("nte-")[-1]}']
                    for hpstr in hpstrs:
                        save_dir = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{tn}.mn-{mn}.{hpstr}.seed-{seed}.pbn-{pbn}'
                        if os.path.exists(f'{save_dir}/test_results.txt'):
                            save_dirs.append(save_dir)

    processes = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    print('# Processes:', processes)
    with Pool(processes=processes) as p:
        with tqdm(total=len(save_dirs)) as pbar:
            for i, _ in enumerate(p.imap_unordered(save_temperature_results, save_dirs)):
                pbar.update()

    print('Done!')


if __name__ == '__main__':
    main()
