#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import vr.programs


def _dataset_to_tensor(dset, mask=None, seed=None, data_slice=None):
  arr = np.asarray(dset, dtype=np.int64)
  if mask is not None:
    arr = arr[mask]
  if seed is not None:
    np.random.seed(seed)
    perm = np.random.permutation(np.arange(arr.shape[0]))
    if data_slice is not None:
      perm = perm[data_slice]
    arr = arr[perm]
  tensor = torch.LongTensor(arr)
  return tensor


class ClevrDataset(Dataset):
  def __init__(self, question_h5, feature_h5, vocab, mode='prefix',
               image_h5=None, max_samples=None, question_families=None,
               image_idx_start_from=None, shuffle=None, data_slice=None):
    mode_choices = ['prefix', 'postfix']
    if mode not in mode_choices:
      raise ValueError('Invalid mode "%s"' % mode)
    self.image_h5 = image_h5
    self.vocab = vocab
    self.feature_h5 = feature_h5
    self.mode = mode
    self.max_samples = max_samples

    mask = None
    if question_families is not None:
      # Use only the specified families
      all_families = np.asarray(question_h5['question_families'])
      N = all_families.shape[0]
      print(question_families)
      target_families = np.asarray(question_families)[:, None]
      mask = (all_families == target_families).any(axis=0)
    if image_idx_start_from is not None:
      all_image_idxs = np.asarray(question_h5['image_idxs'])
      mask = all_image_idxs >= image_idx_start_from

    # Data from the question file is small, so read it all into memory
    print('Reading question data into memory')
    self.all_types = None
    if 'types' in question_h5:
      self.all_types = _dataset_to_tensor(question_h5['types'], mask, shuffle, data_slice)
    self.all_question_families = None
    if 'question_families' in question_h5:
      self.all_question_families = _dataset_to_tensor(question_h5['question_families'], mask, shuffle, data_slice)
    self.all_questions = _dataset_to_tensor(question_h5['questions'], mask, shuffle, data_slice)
    print('Loaded questions: size =', self.all_questions.size())
    self.all_image_idxs = _dataset_to_tensor(question_h5['image_idxs'], mask, shuffle, data_slice)
    self.all_programs = None
    if 'programs' in question_h5:
      self.all_programs = _dataset_to_tensor(question_h5['programs'], mask, shuffle, data_slice)
    self.all_answers = None
    if 'answers' in question_h5:
      self.all_answers = _dataset_to_tensor(question_h5['answers'], mask, shuffle, data_slice)

  def __getitem__(self, index):
    if self.all_question_families is not None:
      question_family = self.all_question_families[index]
    q_type = None if self.all_types is None else self.all_types[index]
    question = self.all_questions[index]
    image_idx = self.all_image_idxs[index]
    answer = None
    if self.all_answers is not None:
      answer = self.all_answers[index]
    program_seq = None
    if self.all_programs is not None:
      program_seq = self.all_programs[index]

    image = None
    if self.image_h5 is not None:
      image = self.image_h5['images'][image_idx]
      image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

    feats = self.feature_h5['features'][image_idx]
    feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

    program_json = None
    if program_seq is not None:
      program_json_seq = []
      for fn_idx in program_seq:
        fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
        if fn_str == '<START>' or fn_str == '<END>': continue
        fn = vr.programs.str_to_function(fn_str)
        program_json_seq.append(fn)
      if self.mode == 'prefix':
        program_json = vr.programs.prefix_to_list(program_json_seq)
      elif self.mode == 'postfix':
        program_json = vr.programs.postfix_to_list(program_json_seq)

    if q_type is None:
      return (question, image, feats, answer, program_seq, program_json)
    return ([question, q_type], image, feats, answer, program_seq, program_json)

  def __len__(self):
    if self.max_samples is None:
      return self.all_questions.size(0)
    else:
      return min(self.max_samples, self.all_questions.size(0))


class ClevrDataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'question_h5' not in kwargs:
      raise ValueError('Must give question_h5')
    if 'feature_h5' not in kwargs:
      raise ValueError('Must give feature_h5')
    if 'vocab' not in kwargs:
      raise ValueError('Must give vocab')

    feature_h5_path = kwargs.pop('feature_h5')
    print('Reading features from', feature_h5_path)
    self.feature_h5 = h5py.File(feature_h5_path, 'r')

    self.image_h5 = None
    if 'image_h5' in kwargs:
      image_h5_path = kwargs.pop('image_h5')
      print('Reading images from ', image_h5_path)
      self.image_h5 = h5py.File(image_h5_path, 'r')

    vocab = kwargs.pop('vocab')
    mode = kwargs.pop('mode', 'prefix')
    shuffle = kwargs.pop('shuffle', None)

    question_families = kwargs.pop('question_families', None)
    max_samples = kwargs.pop('max_samples', None)
    question_h5_path = kwargs.pop('question_h5')
    image_idx_start_from = kwargs.pop('image_idx_start_from', None)
    data_slice = kwargs.pop('data_slice', None)
    print('Reading questions from ', question_h5_path)
    with h5py.File(question_h5_path, 'r') as question_h5:
      self.dataset = ClevrDataset(question_h5, self.feature_h5, vocab, mode,
                                  image_h5=self.image_h5,
                                  max_samples=max_samples,
                                  question_families=question_families,
                                  image_idx_start_from=image_idx_start_from,
                                  shuffle=shuffle, data_slice=data_slice)
    kwargs['collate_fn'] = clevr_collate
    super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

  def close(self):
    if self.image_h5 is not None:
      self.image_h5.close()
    if self.feature_h5 is not None:
      self.feature_h5.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  question_batch = default_collate(transposed[0])
  image_batch = transposed[1]
  if any(img is not None for img in image_batch):
    image_batch = default_collate(image_batch)
  feat_batch = transposed[2]
  if any(f is not None for f in feat_batch):
    feat_batch = default_collate(feat_batch)
  answer_batch = transposed[3]
  if transposed[3][0] is not None:
    answer_batch = default_collate(transposed[3])
  program_seq_batch = transposed[4]
  if transposed[4][0] is not None:
    program_seq_batch = default_collate(transposed[4])
  program_struct_batch = transposed[5]
  return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch]
