import json
import math
import numpy as np
import os
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from transformers import glue_processors as processors
from rda_utils import compute_gaussian_nll

splits = ['train']  # 'dev', 'test'
tt2entropy = {}
for task_name in ['esnli.input-raw', 'mnli', 'cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'qnli', 'wnli', 'rte', 'anli.round1', 'anli.round2', 'anli.round3']:
    print(task_name)
    processor = processors[task_name.split('.', 1)[0]]()
    labels = processor.get_labels()
    if len(labels) <= 1:
        float_precision = 32  # for communicating real values

        ys = []
        for split in splits:
            for ex in getattr(processor, f'get_{split}_examples')(f'{os.environ["BASE_DIR"]}/data/{task_name}'):
                ys.append(float(ex.label))
        ys = np.array(ys)
        domain_size = max(ys) - min(ys)
        ldds_offset = math.log(float_precision) - math.log(domain_size)

        mean = ys.mean()
        std = ys.std()
        print(f'\ty: {round(mean, 3)}+/-{round(std, 3)}')
        h_normal = compute_gaussian_nll(ys, mean, std)
        logp_normal = np.log(multivariate_normal.pdf(ys - mean, mean=0, cov=std))
        ldds_normal = math.log(float_precision) - (logp_normal - math.log(1. / domain_size)).mean()
        print(f'\t Normal: h(y) = {h_normal} | H_{float_precision}(y) = {round(ldds_normal / math.log(2), 2)} bits ({ldds_normal} nats)')

        h_uniform = np.log(domain_size)
        print(f'\tUniform: h(y) = {h_uniform} | H_{float_precision}(y) = {round(math.log(float_precision) / math.log(2), 2)} bits ({math.log(float_precision)} nats)')

        ys = ys.reshape(-1, 1)
        gm = GaussianMixture(n_components=6, random_state=1, n_init=100).fit(ys)  # Typical solution with reasonable fit
        # gm = GaussianMixture(n_components=6, random_state=0, n_init=100).fit(ys)  # Rare solution with good fit
        # gm = GaussianMixture(n_components=6, random_state=21).fit(ys)  # Non-peaked solution with worse-than-uniform fit
        logp_gmm = gm.score_samples(ys)
        h_gmm = -logp_gmm.mean()
        ldds_gmm = math.log(float_precision) - (logp_gmm - math.log(1. / domain_size)).mean()
        print(f'\t    GMM: h(y) = {h_gmm} | H_{float_precision}(y) = {round(ldds_gmm / math.log(2), 2)} bits ({ldds_gmm} nats)')
        nll = h_normal
    else:
        labels = processor.get_labels()
        prior_count = 0
        label2count = {label: prior_count for label in labels}
        for split in splits:
            for ex in getattr(processor, f'get_{split}_examples')(f'{os.environ["BASE_DIR"]}/data/{task_name}'):
                label2count[ex.label] += 1
        probs = np.array([label2count[label] / sum(label2count.values()) for label in labels])
        nll = -np.sum(probs * np.log(probs))
    print(f'\tH(y) = {nll}')
    tt2entropy[task_name.split('.', 1)[0]] = nll
print(json.dumps(tt2entropy, indent=4))
