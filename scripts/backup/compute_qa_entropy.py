import json
import numpy as np
import os
import torch
from tqdm.auto import tqdm


DATA_DIR = os.path.join(os.environ["BASE_DIR"], 'data')
task_name = 'hotpot.context-long.num_subas-0.shuffled-1'
starts = []
span_lens = []
qid2startspanlens = {}
for split in ['trainastest']:  # NB: Requires preprocessing train data as if its test data, then renaming 'train' to 'trainastest'
    cached_features_file = f'{DATA_DIR}/{task_name}/cached.split-{split}_loss.mt-longformer.msl-4096.mql-64'

    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
    )

    with open(f'{DATA_DIR}/hotpot-orig/{split}.json') as f:  # Download
        data = json.load(f)

    for ex, d in tqdm(zip(examples, dataset)):
        qid = ex.qas_id.split('.', 1)[0]
        start = d[3].item()
        end = d[4].item()
        span_len = end - start
        starts.append(start)
        span_lens.append(span_len)
        if qid not in qid2startspanlens:
            qid2startspanlens[qid] = []
        qid2startspanlens[qid].append((start, span_len))
    del features, dataset, examples, features_and_dataset

print(f'Start Position: {round(np.mean(starts), 3)}+/-{round(np.std(starts), 3)} | {min(starts)}-{max(starts)}')
print(f'Span Lengths Range: {round(np.mean(span_lens), 3)}+/-{round(np.std(span_lens), 3)} | {min(span_lens)}-{max(span_lens)}')

prior_count = 0  # Use 1 for smoother prior
startspanlen2count = {(start, span_len): prior_count for start in range(min(starts), max(starts) + 1) for span_len in range(min(span_lens), max(span_lens) + 1)}

for qid, startspanlens in qid2startspanlens.items():
    for startspanlen in startspanlens:
        startspanlen2count[startspanlen] += (1. / len(startspanlens))

out_dim = ((1 + max(starts) - min(starts)) * (1 + max(span_lens) - min(span_lens)))
total_expected_counts = int((out_dim * prior_count) + len(qid2startspanlens.keys()))
print('Output Dimension:', out_dim)
print('# Examples:', len(qid2startspanlens.keys()))
print('Total counts:', total_expected_counts)

counts = np.array([startspanlen2count[(start, spanlen)] for start in range(min(starts), max(starts) + 1) for spanlen in range(min(span_lens), max(span_lens) + 1)])
probs = counts / counts.sum()
pos_probs = probs[probs > 0]

assert int(counts.sum()) == total_expected_counts, f'Got {int(counts.sum())} total counts but expected {total_expected_counts}'
nll = -np.sum(pos_probs * np.log(pos_probs))
print(f'H(y) = {nll}')
