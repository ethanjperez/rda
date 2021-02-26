---
language: en
datasets:
- squad
---

# T5-base fine-tuned on SQuAD for **Question Generation**

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) fine-tuned on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) for **Question Generation** by just prepending the *answer* to the *context*.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)


## Details of the downstream task (Q&A) - Dataset 📚 🧐 ❓

Dataset ID: ```squad``` from  [HugginFace/NLP](https://github.com/huggingface/nlp)

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| squad | train | 87599    |
| squad | valid  | 10570    |

How to load it from [nlp](https://github.com/huggingface/nlp)

```python
train_dataset  = nlp.load_dataset('squad, split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
```
Check out more about this dataset and others in [NLP Viewer](https://huggingface.co/nlp/viewer/)


## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this  awesome one](https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb) by [Suraj Patil](https://twitter.com/psuraj28)

He also made a great research on [**Question Generation**](https://github.com/patil-suraj/question_generation)


## Model in Action 🚀

```python
# Tip: By now, install transformers from source

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

context = "Manuel have created RuPERTa-base with the support of HF-Transformers and Google"
answer = "Manuel"

get_question(answer, context)

# output: question: Who created the RuPERTa-base?
```

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
