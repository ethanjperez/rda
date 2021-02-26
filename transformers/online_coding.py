import argparse
from decimal import Decimal
import os
import shutil
import sys
from transformers.loss_data_utils import mn2hps, mn2max_tbs
from time import time
from transformers import run_pl_glue


def main():
    current_filename = sys.argv[0][:]
    print(f'{current_filename}: Starting online coding run')
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mn", default=None, type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tn", default="", type=str, required=True, help="The task name / dataset to run on")
    parser.add_argument("--max_pbn", default=7, type=int, required=True, help="Max prequential block number to include. Use -1 to just preprocess dataset.")
    parser.add_argument("--cpus", default=1, type=int, required=True, help="The number of available CPUs (for e.g. data loading and preprocessing).")
    args = parser.parse_args()

    hps = mn2hps(args.mn)
    assert len(hps['nte']) == 1, 'Expected only 1 NTE in hyperparameters set'
    nte = hps['nte'][0]
    min_bs = min(hps['bs'])
    model_type = args.mn
    if args.mn.endswith('.from_scratch'):
        model_type = args.mn.rsplit('.', 1)[0]
    max_tbs = mn2max_tbs[model_type]
    task_type = args.tn.split('.')[0]

    if args.max_pbn == -1:
        gpus = 0
        flags = '--preprocess_only'
        pbns = [-1]
    else:
        gpus = 1
        flags = "--fp16 --fp16_opt_level O2"
        pbns = list(range(args.max_pbn + 1))

    def launch(bs, lr, seed, pbn):
        # Check that launching a run with these hyperparameters is necessary
        output_dir = f"checkpoint/rda/tn-{args.tn}.mn-{args.mn}.bs-{bs}.lr-{lr}.nte-{nte}.seed-{seed}.pbn-{Decimal.from_float(pbn)}"
        if os.path.exists(f'{os.environ["BASE_DIR"]}/{output_dir}/test_results.txt'):
            print(f'{current_filename}: Skipping bs={bs} lr={lr} seed={seed}: Test file exists: {os.environ["BASE_DIR"]}/{output_dir}/test_results.txt')
            return

        # Remove existing directory if it exists
        if os.path.exists(f'{os.environ["BASE_DIR"]}/{output_dir}'):
            shutil.rmtree(f'{os.environ["BASE_DIR"]}/{output_dir}')
        os.makedirs(f'{os.environ["BASE_DIR"]}/{output_dir}', exist_ok=True)

        # Create python command and launch
        gas = max(((bs - 1) // max_tbs) + 1, 1)
        tbs = bs // gas
        cmd = f'python examples/text-classification/run_pl_glue.py --model_name_or_path {args.mn} --output_dir {output_dir} --task {task_type} --do_train --do_predict --data_dir data/{args.tn} --max_seq_length 512 --train_batch_size {tbs} --eval_batch_size {tbs * 2} --learning_rate {lr} --num_train_epochs {nte} --seed {seed} --gradient_accumulation_steps {gas} --criterion min_val_loss --patience 0 {flags} --gpus {gpus} --prequential_block_no {pbn} --prequential_num_blocks 8 --num_workers {args.cpus}'
        if 'roberta' in args.mn:
            cmd += ' --adam_beta_2 0.98 --adam_epsilon 1e-6 --max_grad_norm inf --warmup_proportion 0.06 --weight_decay 0.1'
        elif 'gpt' in args.mn:
            cmd += ' --warmup_proportion 0.002 --weight_decay 0.01'
        elif 'facebook/bart' in args.mn:
            cmd += ' --adam_beta_2 0.98 --max_grad_norm inf --warmup_proportion 0.06 --weight_decay 0.01'
        elif ('albert' in args.mn) or ('xlnet' in args.mn):
            cmd += ' --adam_epsilon 1e-6 --warmup_proportion 0.1 --weight_decay 0.01'
        else:
            NotImplementedError(f'{current_filename}: Model Name = {args.mn}')
        print(f'{current_filename}: Running command: {cmd}')
        sys.argv = cmd.split()[1:]
        run_pl_glue.main()
        return

    for pbn in pbns:
        if pbn == -1:
            print(f'{current_filename}: Preprocessing')
            launch(min_bs, 1e-6, 12, pbn)
        else:
            print(f'{current_filename}: Running HP Sweep')
            for batch_size in hps['bs']:
                for learning_rate in hps['lr']:
                    launch(batch_size, learning_rate, 12, pbn)
            print(f'{current_filename}: Running seeds for best HPs')
            for random_seed in [20, 21, 22, 23]:
                launch(0, 0, random_seed, pbn)
    print(f'{current_filename}: Finished sweep in {round(time() - start_time)}s')


if __name__ == "__main__":
    main()
