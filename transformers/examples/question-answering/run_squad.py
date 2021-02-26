# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import collections
import glob
import json
import logging
import os
import random
import shutil
from time import time, sleep
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from lightning_base import remove_extra_padding
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, features2dataset
from transformers.loss_data_utils import mn2hpstrs, read_results


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer, tb_writer, all_train_dataset_examples_features):
    """ Train the model """
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if global_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.save_freq is not None:
        args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps // args.save_freq

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta_1, args.adam_beta_2),
                      eps=args.adam_epsilon)
    num_warmup_steps = (args.warmup_proportion * t_total) if args.warmup_steps is None else args.warmup_steps
    if args.learning_rate_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    elif args.learning_rate_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    else:
        raise NotImplementedError('LR Schedule Type: ' + args.learning_rate_schedule)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16_opt_level >= 0:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level='O' + str(args.fp16_opt_level))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    effective_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
                                 (torch.distributed.get_world_size() if global_rank != -1 else 1)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", effective_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    best_eval_loss = float('inf')
    best_global_examples_seen = 0
    early_stop = False
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=global_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        tr_batch_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=global_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            remove_extra_padding(batch)
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16_opt_level >= 0:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_batch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16_opt_level >= 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_examples_seen = global_step * effective_train_batch_size
                global_step += 1

                epoch_iterator.desc = 'loss: {:.2e} lr: {:.2e}'.format(tr_batch_loss, scheduler.get_lr()[0])
                tr_batch_loss = 0.0
                # Log metrics
                if global_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_examples_seen)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_examples_seen)
                    logging_loss = tr_loss

                # Save model checkpoint
                if global_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_examples_seen))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if global_rank == -1:
                        results = evaluate(args, model, tokenizer, prefix=global_examples_seen, split='dev', all_train_dataset_examples_features=all_train_dataset_examples_features)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_examples_seen)
                        if results['loss'] <= best_eval_loss:
                            best_eval_loss = results['loss']
                            best_global_examples_seen = global_examples_seen
                        else:
                            early_stop = True
                            break

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if early_stop or (args.max_steps > 0 and global_step > args.max_steps):
            train_iterator.close()
            break

    if global_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_eval_loss, best_global_examples_seen


def evaluate(args, model, tokenizer, prefix="", split='test', all_train_dataset_examples_features=None):
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()
    dataset, examples, features = load_and_cache_examples(args, tokenizer, split=split, output_examples=True, all_train_dataset_examples_features=all_train_dataset_examples_features)

    if not os.path.exists(args.output_dir) and global_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        remove_extra_padding(batch)
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3] if args.evaluate_loss else None,
                "end_positions": batch[4] if args.evaluate_loss else None,
                "reduction": "none",
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[8 if args.evaluate_loss else 3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5 if args.evaluate_loss else 4], "p_mask": batch[6 if args.evaluate_loss else 5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            loss = None
            if args.evaluate_loss:
                loss = output[0]
                output = output[1:]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    loss=loss,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits, loss=loss)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        assert not args.evaluate_loss, '--evaluate_loss not implemented for xlnet and xlm'
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        avg_loss = float('inf')
        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions, avg_loss = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    eval_predictions = collections.OrderedDict()
    for k, v in predictions.items():
        if k.endswith('.0') or ('.' not in k):
            eval_predictions[k] = v
    eval_examples = [x for x in examples if (x.qas_id.endswith('.0') or ('.' not in x.qas_id))]
    results = squad_evaluate(eval_examples, eval_predictions)
    results['loss'] = avg_loss
    return results


def load_and_cache_examples(args, tokenizer, split='train', output_examples=False, output_full_features_and_dataset=False, all_train_dataset_examples_features=None):
    evaluate = split != 'train'
    get_loss = not (evaluate and not args.evaluate_loss)
    global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()
    if global_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split2input_file = {
        'train': args.train_file,
        'dev': args.train_file if args.prequential else args.dev_file,
        'test': args.train_file if args.prequential else args.test_file
    }
    input_file = os.path.join(input_dir, split2input_file[split])
    cache_filename = 'cached.split-{}.mt-{}.msl-{}.mql-{}'.format(
        '.'.join(input_file.split('/')[-1].split('.')[:-1]) + ('_loss' if args.evaluate_loss else ''),
        args.tokenizer_name if args.tokenizer_name else args.model_type,
        str(args.max_seq_length),
        str(args.max_query_length))
    cached_features_file = os.path.join(os.path.dirname(input_file), cache_filename)
    print('Cached features file {} exists: {}'.format(cached_features_file, str(os.path.exists(cached_features_file))))
    os.makedirs(cached_features_file.rsplit('/', 1)[0], exist_ok=True)

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and args.prequential and (all_train_dataset_examples_features is not None):
        logger.info("Already loaded features from cached file %s", cached_features_file)
        dataset, examples, features = all_train_dataset_examples_features
    elif os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (((split == 'test') and not args.test_file) or ((split == 'dev') and not args.dev_file) or ((split == 'train') and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate and (not args.prequential):
                examples = processor.get_dev_examples(args.data_dir, filename=split2input_file[split], use_all_answers=False)  # use_all_answers=args.evaluate_loss to marginalize over possible targets
            else:
                examples = processor.get_train_examples(args.data_dir, filename=split2input_file[split])

        features, dataset, examples = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=get_loss,
            return_dataset="pt",
            threads=args.threads,
        )

        if global_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    assert len(dataset) == len(examples)
    logger.info(f'# Valid Total Examples: {len(dataset)} ({split} set)')

    prequential_max_num_train_samples = len(features) if args.prequential_max_block_size == -1 else min(args.prequential_max_block_size, len(features))
    idxs_slice = None
    random_slice = True
    new_dataset, new_examples, new_features = dataset, examples, features
    if args.prequential:
        block_start_idxs = np.round(np.logspace(np.log10(args.prequential_min_block_size), np.log10(prequential_max_num_train_samples), args.prequential_num_blocks + 1)).astype(int)
        num_train_samples = int(round((1. - args.prequential_valid_frac) * block_start_idxs[args.prequential_block_no]))
        mode2slice = {
            'train': slice(None, num_train_samples),
            'dev': slice(num_train_samples, block_start_idxs[args.prequential_block_no]),
            'test': slice(block_start_idxs[args.prequential_block_no], block_start_idxs[args.prequential_block_no + 1]),
        }
        assert split in {'train', 'dev', 'test'}, f'Unexpected split = {split} not in ["train", "dev", "test"]'
        idxs_slice = mode2slice[split]
    elif args.prequential_block_no == args.prequential_num_blocks:
        if not evaluate:
            idxs_slice = slice(None, prequential_max_num_train_samples)

    if idxs_slice is not None:
        idxs = list(range(len(features)))
        if random_slice:
            random.Random(args.seed).shuffle(idxs)
        idxs = idxs[idxs_slice]
        new_features = [features[idx] for idx in idxs]
        new_dataset = features2dataset(new_features, get_loss)
        new_examples = [examples[idx] for idx in idxs]

    assert len(new_dataset) == len(new_examples)
    logger.info(f'# Valid Examples Used: {len(new_dataset)} ({split} set)')

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    assert not (output_examples and output_full_features_and_dataset), 'Cannot use both output_examples and output_features_and_dataset'
    if output_full_features_and_dataset:
        return new_dataset, (dataset, examples, features)
    if output_examples:
        return new_dataset, new_examples, new_features
    return new_dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--from_scratch", action="store_true", help="If true, train model weights from scratch.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default='train.json',
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--dev_file",
        default='dev.json',
        type=str,
        help="The input dev file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--test_file",
        default='test.json',
        type=str,
        help="The input test file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_loss", action="store_true", help="Whether to evaluate loss on the dev set.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--preprocess_only", action='store_true', help="Whether to only run preprocessing.")
    parser.add_argument("--prequential_block_no", default=None, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train). "
                        "Use -1 for --preprocess_only and --prequential_num_blocks value to train on all prequential code examples and evaluate on test.")
    parser.add_argument("--prequential_num_blocks", default=8, type=int, help="The number of blocks sent when calculating prequential code length (number of models to train).")
    parser.add_argument("--prequential_min_block_size", default=64, type=int, help="The minimum number of examples to use in a training block (inclusive).")
    parser.add_argument("--prequential_max_block_size", default=-1, type=int, help="The maximum number of examples to use in a training block (exclusive). -1 for all examples.")
    parser.add_argument("--prequential_valid_frac", default=0.1, type=float, help="The fraction of examples to use for validation.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_schedule", default="linear", type=str, help="The type of learning rate schedule.",
                        choices=["linear", "constant"])
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta_1", default=0.9, type=float, help="Beta 1 for Adam optimizer.")
    parser.add_argument("--adam_beta_2", default=0.999, type=float, help="Beta 2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--early_stopping", action="store_true", help="If true, then early stop based on validation loss.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument("--warmup_steps", default=None, type=int, help="Linear warmup over warmup_steps. Overrides --warmup_proportion")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_freq', type=int, default=1, help="Save/evaluate X times per epoch. Overrides save_steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16_opt_level",
        type=int,
        default=-1,  # fp32
        help="For fp16: Apex AMP optimization level selected in [0, 1, 2, 3]."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
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
    if args.learning_rate == 0:
        assert args.data_dir is not None, '--learning_rate 0 requires --data_dir'
        task_name = args.data_dir.split('data/', 1)[-1]
        hpstrs = mn2hpstrs(args.model_name_or_path)
        best_hpstr = None
        best_val_loss = float('inf')
        for hpstr in hpstrs:
            hp_sweep_seed = 12  # NB: Change to use a different seed for HP sweep
            save_dir = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{task_name}.mn-{args.model_name_or_path}.{hpstr}.seed-{hp_sweep_seed}.pbn-{args.prequential_block_no}'
            val_stats = read_results(save_dir, read_json=True)
            print(f'{hpstr}\tVal={round(val_stats["val_loss"], 3)}')
            if val_stats['val_loss'] < best_val_loss:
                best_val_loss = val_stats['val_loss']
                best_hpstr = hpstr

        print(f'Best: {best_hpstr} Val={round(best_val_loss, 3)}\n')
        assert best_hpstr is not None, f'No validation results read from previous runs!'
        args.learning_rate = float(best_hpstr.split('lr-', 1)[-1].split('.nte')[0])
        print(f'Loaded LR={args.learning_rate}')

    if args.model_name_or_path.endswith('.from_scratch'):
        args.model_name_or_path = args.model_name_or_path.rsplit('.', 1)[0]
        args.from_scratch = True

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    print('-' * 40, 'START', '-' * 40, args.local_rank)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        global_rank = -1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_PORT'] = str(args.master_port)
        print('MASTER_PORT:', os.environ['MASTER_PORT'])
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        global_rank = torch.distributed.get_rank()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if global_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process local rank: %s, global rank %s/%s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training opt level: %s",
        args.local_rank,
        global_rank,
        (torch.distributed.get_world_size() if global_rank != -1 else -1),
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16_opt_level,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if global_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        from_scratch=args.from_scratch,
    )

    if global_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Set up tensorboard metric logging
    if not args.preprocess_only:
        tb_writer = SummaryWriter(args.output_dir) if global_rank in [-1, 0] else None

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16_opt_level >= 0:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    best_eval_loss = float('inf')
    best_global_examples_seen = 0
    all_train_dataset_examples_features = None
    if args.do_train:
        train_dataset, all_train_dataset_examples_features = load_and_cache_examples(args, tokenizer, split='train', output_full_features_and_dataset=True)
        if args.preprocess_only:
            load_and_cache_examples(args, tokenizer, split='dev', output_examples=False)
            load_and_cache_examples(args, tokenizer, split='test', output_examples=False)
            return
        else:
            global_step, tr_loss, best_eval_loss, best_global_examples_seen = train(args, train_dataset, model, tokenizer, tb_writer, all_train_dataset_examples_features)
            logger.info("\n\n global_step = %s, average loss = %s\n\n", global_step, tr_loss)

            # Save the trained model and the tokenizer
            if global_rank in [-1, 0]:
                # Create output directory if needed
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                # Load a trained model and vocabulary that you have fine-tuned
                model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
                model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {'val_loss': best_eval_loss, 'best_global_examples_seen': best_global_examples_seen}
    logger.info(f'\n\nBest val loss: {best_eval_loss} / best_global_examples_seen: {best_global_examples_seen}\n\n')
    if (global_rank == 0) or (args.do_eval and global_rank == -1):
        if args.preprocess_only:
            load_and_cache_examples(args, tokenizer, split='test', output_examples=False)
            return

        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [os.path.join(args.output_dir, 'checkpoint-' + str(best_global_examples_seen))]  # NB: To eval all checkpoints, use: [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_examples_seen = checkpoint.split('-')[-1]
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            prefix = args.test_file.split('/')[-1].rsplit('.', 1)[0]
            result = evaluate(args, model, tokenizer, prefix=f'{prefix}-{global_examples_seen}', split='test', all_train_dataset_examples_features=all_train_dataset_examples_features)

            logger.info(f'\n\nTEST LOSS: {result["loss"]}\n\n')
            for key, value in results.items():
                tb_writer.add_scalar('test_{}'.format(key), value, global_examples_seen)
            if str(best_global_examples_seen) == str(global_examples_seen):
                results['avg_test_loss'] = result['loss']
                results['test_loss'] = result['loss']
            result = dict((k + ('_{}'.format(global_examples_seen) if global_examples_seen else ''), v) for k, v in result.items())
            results.update(result)

        logger.info("Results: {}".format(results))
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

        # Delete all unnecessary files
        for path in os.listdir(args.output_dir):
            if (f'_{best_global_examples_seen}.json' in path) or (f'-{best_global_examples_seen}.json' in path) or (path == 'results.json') or (path == 'train.log'):
                continue
            path = os.path.join(args.output_dir, path)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError as e:  # In case "Directory not empty" if a file was written since deletion attempt
                    sleep(30)
                    shutil.rmtree(path)
        # # Simplify names of files for best epoch model
        # for filename in os.listdir(args.output_dir):
        #     new_filename = filename.replace(f'-{best_global_examples_seen}', '').replace(f'_{best_global_examples_seen}', '')
        #     if new_filename != filename:
        #         os.rename(os.path.join(args.output_dir, filename), os.path.join(args.output_dir, new_filename))

    return results


if __name__ == "__main__":
    start = time()
    main()
    print(f'Finished training!! Time (s): {round(time() - start)}')
