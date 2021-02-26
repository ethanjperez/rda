# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
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
""" Tokenization class for model T5."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from .file_utils import add_start_docstrings, is_sentencepiece_available
from .tokenization_utils import BatchEncoding
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .utils import logging


if is_sentencepiece_available():
    from .tokenization_t5 import T5Tokenizer
else:
    T5Tokenizer = None


logger = logging.get_logger(__name__)

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to file names for serializing Tokenizer instances
####################################################
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to pretrained vocabulary URL for all the model shortcut names.
####################################################
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    },
    "tokenizer_file": {
        "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-tokenizer.json",
        "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-tokenizer.json",
        "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-tokenizer.json",
        "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-tokenizer.json",
        "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-tokenizer.json",
    },
}

####################################################
# Mapping from model shortcut names to max length of inputs
####################################################
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class T5TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" T5 tokenizer (backed by HuggingFace's `tokenizers` library). Based on `SentencePiece
    <https://github.com/google/sentencepiece>`__ .

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end
                of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (:obj:`int`, `optional`, defaults to 100):
            Add a number of extra ids added to the end of the vocabulary for use as sentinels.
            These tokens are accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1.
            Extra tokens are indexed from the end of the vocabulary up to beginnning ("<extra_id_0>" is the last token
            in the vocabulary like in T5 preprocessing see `here
            <https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117>`__).
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]
    slow_tokenizer_class = T5Tokenizer

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        if extra_ids > 0:
            all_extra_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            if all(tok not in self.additional_special_tokens for tok in all_extra_tokens):
                self.additional_special_tokens = self.additional_special_tokens + [
                    "<extra_id_{}>".format(i) for i in range(extra_ids)
                ]

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A sequence has the following format:

        - single sequence: ``X </s>``
        - pair of sequences: ``A </s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.max_len
        self.prefix_tokens = []
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        # set prefix_tokens for target text
        self.prefix_tokens = [self.pad_token_id]
        labels_and_decoder_mask = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )
        model_inputs["labels"] = labels_and_decoder_mask["input_ids"]
        self.prefix_tokens = []
        return model_inputs
