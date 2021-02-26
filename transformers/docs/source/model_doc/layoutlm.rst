LayoutLM
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The LayoutLM model was proposed in the paper `LayoutLM: Pre-training of Text and Layout for Document Image Understanding <https://arxiv.org/abs/1912.13318>`__
by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. It's a simple but effective pre-training method
of text and layout for document image understanding and information extraction tasks, such as form understanding and receipt understanding.

The abstract from the paper is the following:

*Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the widespread use of pre-training models for NLP applications, they almost exclusively focus on text-level manipulation, while neglecting layout and style information that is vital for document image understanding. In this paper, we propose the \textbf{LayoutLM} to jointly model interactions between text and layout information across scanned document images, which is beneficial for a great number of real-world document image understanding tasks such as information extraction from scanned documents. Furthermore, we also leverage image features to incorporate words' visual information into LayoutLM. To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for document-level pre-training. It achieves new state-of-the-art results in several downstream tasks, including form understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification (from 93.07 to 94.42).*

Tips:

- LayoutLM has an extra input called :obj:`bbox`, which is the bounding boxes of the input tokens.
- The :obj:`bbox` requires the data that on 0-1000 scale, which means you should normalize the bounding box before passing them into model.

The original code can be found `here <https://github.com/microsoft/unilm/tree/master/layoutlm>`_.


LayoutLMConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMConfig
    :members:


LayoutLMTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMTokenizer
    :members:


LayoutLMModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMModel
    :members:


LayoutLMForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMForMaskedLM
    :members:


LayoutLMForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMForTokenClassification
    :members:
