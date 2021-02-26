import argparse
import glob
import logging
import os
import random
import shutil
from time import sleep
from tqdm import tqdm
from argparse import Namespace

from decimal import Decimal
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_base import BaseTransformer, add_generic_args, generic_train, DynamicPaddingCallback
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes
from transformers import glue_processors as processors
from transformers import glue_tasks_num_labels
from transformers.loss_data_utils import mn2hpstrs, mn2max_tbs, read_results

logger = logging.getLogger(__name__)


class GLUETransformer(BaseTransformer):

    mode = "sequence-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.glue_output_mode = glue_output_modes[hparams.task]
        num_labels = glue_tasks_num_labels[hparams.task]
        self.best_val_loss = float('inf')
        self.best_val_epoch = None

        super().__init__(hparams, num_labels, self.mode)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type not in ["distilbert", "bart"]:
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss = outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss": loss.detach().cpu(), "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log": tensorboard_logs}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        processor = processors[args.task]()
        self.labels = processor.get_labels()

        for mode in ['train'] if self.hparams.prequential else ['train', 'dev', 'test']:
            cached_features_file = self._feature_file(mode)
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if mode == "dev"
                    else (processor.get_test_examples(args.data_dir) if mode == "test" else
                          processor.get_train_examples(args.data_dir))
                )
                features = convert_examples_to_features(
                    examples,
                    self.tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.labels,
                    output_mode=args.glue_output_mode,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.preprocess_only:
            exit('Done preprocessing! Exiting now.')

    def get_dataloader(self, mode: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        "Load datasets. Called after prepare data."

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        # mode = "dev" if mode == "test" else mode

        cached_features_file = self._feature_file('train' if self.hparams.prequential else mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        prequential_max_num_train_samples = len(features) if self.hparams.prequential_max_block_size == -1 else min(self.hparams.prequential_max_block_size, len(features))
        if self.hparams.prequential:
            random.Random(self.hparams.seed).shuffle(features)
            block_start_idxs = np.round(np.logspace(np.log10(self.hparams.prequential_min_block_size), np.log10(prequential_max_num_train_samples), self.hparams.prequential_num_blocks + 1)).astype(int)
            num_train_samples = int(round((1. - self.hparams.prequential_valid_frac) * block_start_idxs[self.hparams.prequential_block_no]))
            mode2slice = {
                'train': slice(None, num_train_samples),
                'dev': slice(num_train_samples, block_start_idxs[self.hparams.prequential_block_no]),
                'test': slice(block_start_idxs[self.hparams.prequential_block_no], block_start_idxs[self.hparams.prequential_block_no + 1]),
            }
            assert mode in {'train', 'dev', 'test'}, f'Unexpected mode = {mode} not in ["train", "dev", "test"]'
            features = features[mode2slice[mode]]
        elif self.hparams.prequential_block_no == self.hparams.prequential_num_blocks:
            if mode == 'train':
                random.Random(self.hparams.seed).shuffle(features)
                features = features[:prequential_max_num_train_samples]
        all_input_ids = torch.tensor([f.input_ids for f in tqdm(features, desc=f'Loading {mode} all_input_ids...')], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in tqdm(features, desc=f'Loading {mode} all_attention_mask...')], dtype=torch.long)
        all_token_type_ids = [f.token_type_ids for f in tqdm(features, desc=f'Loading {mode} all_token_type_ids...')]
        if all(tti is None for tti in all_token_type_ids):
            all_token_type_ids = [0 for _ in tqdm(features)]
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        if self.hparams.glue_output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.hparams.glue_output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        logger.info(f"Loaded {len(features)} {mode} examples!")

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers  # NB: Set to 0 if causing issues
        )

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        if self.config.model_type not in ["distilbert", "bart"]:
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        val_loss = tmp_eval_loss.detach().cpu()
        return {"val_loss": val_loss, "pred": preds, "target": out_label_ids, "log": {"loss": val_loss}, "progress_bar": {"loss": val_loss}}

    def _eval_end(self, outputs, save_name=None) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        preds_full = np.concatenate([x["pred"] for x in outputs], axis=0)

        if self.hparams.glue_output_mode == "classification":
            preds = np.argmax(preds_full, axis=1)
            preds_full = softmax(preds_full, axis=1)
        elif self.hparams.glue_output_mode == "regression":
            preds = np.squeeze(preds_full, axis=1)
        else:
            preds = preds_full

        # Save dev/test predictions to file
        if save_name is not None:
            targets = np.concatenate([x["target"] for x in outputs], axis=0)
            os.makedirs(self.hparams.output_dir, exist_ok=True)
            with open(os.path.join(self.hparams.output_dir, f'preds.{save_name}.csv'), 'w') as f:
                f.write(','.join([f'p({str(p)})' for p in range(len(preds_full[0].tolist()))] + ['label']) + '\n')
                for pred_full, target in zip(preds_full, targets):
                    f.write(','.join([str(p) for p in pred_full.tolist()] + [str(target)]) + '\n')

        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **compute_metrics(self.hparams.task, preds, out_label_ids)}

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs, save_name=f'val.epoch-{self.current_epoch}')
        logs = ret["log"]
        cur_val_loss = logs["val_loss"].detach().cpu().item()
        if cur_val_loss < self.best_val_loss:
            self.best_val_epoch = self.current_epoch
        self.best_val_loss = min(cur_val_loss, self.best_val_loss)
        logger.info(f'Best Val Loss So Far: {self.best_val_loss}')
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs, f'test.epoch-{self.current_epoch}')
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        logs = {f'test_{k.split("val_", 1)[-1]}': v for k, v in ret["log"].items()}
        return {"avg_test_loss": logs["test_loss"].item(), "best_val_loss": self.best_val_loss, "best_val_epoch": self.best_val_epoch, "log": logs, "progress_bar": logs}

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument("--prequential_block_no", default=None, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train). "
                            "Use -1 for --preprocess_only and --prequential_num_blocks value to train on all prequential code examples and evaluate on test.")
        parser.add_argument("--prequential_num_blocks", default=8, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train).")
        parser.add_argument("--prequential_min_block_size", default=64, type=int, help="The minimum number of examples to use in a training block (inclusive).")
        parser.add_argument("--prequential_max_block_size", default=10000, type=int, help="The maximum number of examples to use in a training block (exclusive). -1 for all examples.")
        parser.add_argument("--prequential_valid_frac", default=0.1, type=float, help="The fraction of examples to use for validation.")

        parser.add_argument("--preprocess_only", action='store_true', help="Preprocess and cache dataset, then exit.")
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--task",
            default="",
            type=str,
            required=True,
            help="The GLUE task to run",
        )
        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        return parser


def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = GLUETransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    args.prequential = False
    if args.prequential_block_no is not None:
        assert args.prequential_block_no <= args.prequential_num_blocks, f'Expected --prequential_block_no ({args.prequential_block_no}) <= --prequential_num_blocks ({args.prequential_num_blocks})'
        assert args.prequential_block_no >= -1, f'Expected --prequential_block_no ({args.prequential_block_no}) >= -1'
        assert args.prequential_num_blocks >= 1, f'Expected --prequential_num_blocks ({args.prequential_num_blocks}) >= 1'
        if args.prequential_block_no == args.prequential_num_blocks:  # Train on all, val with held-in dev, test on held-out dev
            pass
        elif args.prequential_block_no == -1:  # Preprocess
            args.preprocess_only = True
        else:
            args.prequential = True
    if args.preprocess_only:
        args.do_train = True
    task_name = args.data_dir.split('data/', 1)[-1]
    if (args.train_batch_size == 0) and (args.learning_rate == 0):
        num_missing_files = 0
        hpstrs = mn2hpstrs(args.model_name_or_path)
        best_hpstr = None
        best_val_loss = float('inf')
        for hpstr in hpstrs:
            save_dir = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{task_name}.mn-{args.model_name_or_path}.{hpstr}.seed-12.pbn-{args.prequential_block_no}'
            try:
                val_filename = f'{save_dir}/val_results.txt'
                val_stats = read_results(save_dir, val_filename) if os.path.exists(val_filename) else read_results(save_dir)
                print(f'{hpstr}\tVal={round(val_stats["val_loss"], 3)}')
                if val_stats['val_loss'] < best_val_loss:
                    best_val_loss = val_stats['val_loss']
                    best_hpstr = hpstr
            except FileNotFoundError as e:
                print(f'\tFileNotFoundError for {save_dir}')
                num_missing_files += 1

        print(f'Best: {best_hpstr} Val={round(best_val_loss, 3)}\n')
        print(f'# Missing Files: {num_missing_files}\n')
        assert best_hpstr is not None, f'No validation results read from previous runs!'
        effective_batch_size = int(best_hpstr.split('bs-', 1)[-1].split('.')[0])
        model_type = args.model_name_or_path
        if args.model_name_or_path.endswith('.from_scratch'):
            model_type = args.model_name_or_path.rsplit('.', 1)[0]
        max_tbs = mn2max_tbs[model_type]
        args.accumulate_grad_batches = ((effective_batch_size - 1) // max_tbs) + 1
        args.train_batch_size = effective_batch_size // args.accumulate_grad_batches
        args.eval_batch_size = 2 * args.train_batch_size
        args.learning_rate = float(best_hpstr.split('lr-', 1)[-1].split('.nte')[0])
        print(f'Loaded GAS={args.accumulate_grad_batches} | TBS={args.train_batch_size} | BS={effective_batch_size} | LR={args.learning_rate}')

    if args.model_name_or_path.endswith('.from_scratch'):
        args.model_name_or_path = args.model_name_or_path.rsplit('.', 1)[0]
        args.from_scratch = True

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        effective_batch_size = args.train_batch_size * args.accumulate_grad_batches * max(1, args.gpus)
        args.output_dir = f"./checkpoint/rda/tn-{task_name}.mn-{args.model_name_or_path}.lr-{args.learning_rate}.bs-{effective_batch_size}.nte-{args.num_train_epochs}.seed-{args.seed}.pbn-{args.prequential_block_no}"
        os.makedirs(args.output_dir)

    model = GLUETransformer(args)
    trainer = generic_train(model, args, extra_callbacks=[DynamicPaddingCallback()])
    best_val_loss = model.best_val_loss
    best_val_epoch = model.best_val_epoch
    logger.info(f'Best Val Loss: {best_val_loss} @ Epoch {best_val_epoch}')

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.best_val_loss = best_val_loss
        model.best_val_epoch = best_val_epoch
        result = trainer.test(model=model, test_dataloaders=model.test_dataloader())
        # Remove unnecessary files
        os.remove(checkpoints[-1])
        shutil.rmtree(os.path.join(args.output_dir, 'best_tfmr'))
        for dirpath in glob.glob(os.path.join(args.output_dir, "version_*"), recursive=False):
            try:
                shutil.rmtree(dirpath)
            except OSError as e:  # In case "Directory not empty" if a file was written since deletion attempt
                sleep(30)
                shutil.rmtree(dirpath)
        if model.best_val_epoch is not None:
            for predspath in glob.glob(os.path.join(args.output_dir, "preds.val.epoch-*.csv"), recursive=False):
                if not predspath.endswith(f"preds.val.epoch-{model.best_val_epoch}.csv"):
                    os.remove(predspath)
        return result


if __name__ == "__main__":
    main()
