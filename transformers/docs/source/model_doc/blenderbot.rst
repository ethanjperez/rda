Blenderbot
-----------------------------------------------------------------------------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ .

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Blender chatbot model was proposed in `Recipes for building an open-domain chatbot <https://arxiv.org/pdf/2004.13637.pdf>`__ Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

*Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that scaling neural models in the number of parameters and the size of the data they are trained on gives improved results, we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent persona. We show that large scale models can learn these skills when given appropriate training data and choice of generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing failure cases of our models.*

The authors' code can be found `here <https://github.com/facebookresearch/ParlAI>`__ .


Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Blenderbot uses a standard `seq2seq model transformer <https://arxiv.org/pdf/1706.03762.pdf>`__ based architecture.
- It inherits completely from :class:`~transformers.BartForConditionalGeneration`
- Even though blenderbot is one model, it uses two tokenizers :class:`~transformers.BlenderbotSmallTokenizer` for 90M checkpoint and :class:`~transformers.BlenderbotTokenizer` for all other checkpoints.
- :class:`~transformers.BlenderbotSmallTokenizer` will always return :class:`~transformers.BlenderbotSmallTokenizer`, regardless of checkpoint. To use the 3B parameter checkpoint, you must call :class:`~transformers.BlenderbotTokenizer` directly.
- Available checkpoints can be found in the `model hub <https://huggingface.co/models?search=blenderbot>`__.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model Usage:

        >>> from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
        >>> mname = 'facebook/blenderbot-90M'
        >>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
        >>> UTTERANCE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([UTTERANCE], return_tensors='pt')
        >>> reply_ids = model.generate(**inputs)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])


See Config Values:

        >>> from transformers import BlenderbotConfig
        >>> config_90 = BlenderbotConfig.from_pretrained("facebook/blenderbot-90M")
        >>> config_90.to_diff_dict()  # show interesting Values.
        >>> configuration_3B = BlenderbotConfig("facebook/blenderbot-3B")
        >>> configuration_3B.to_diff_dict()


BlenderbotConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: transformers.BlenderbotConfig
    :members:

BlenderbotTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotTokenizer
    :members: build_inputs_with_special_tokens

BlenderbotSmallTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotSmallTokenizer
    :members:


BlenderbotForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
See :obj:`transformers.BartForConditionalGeneration` for arguments to `forward` and `generate`

.. autoclass:: transformers.BlenderbotForConditionalGeneration
    :members:
