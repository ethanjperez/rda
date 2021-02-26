import json
import math
import numpy as np
import os


group2answers = {
    'number': [str(i) for i in range(11)],
    'color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'shape': ['cube', 'cylinder', 'sphere'],
    'size': ['large', 'small'],
    'material': ['metal', 'rubber'],
    'boolean': ['yes', 'no']
}
tt2entropy = {}
for template_type in ['comparison', 'compare_integer', 'same_relate']:
    with open(f'{os.environ["BASE_DIR"]}/data/CLEVR_v1.0/questions.{template_type}.num_sas-0/CLEVR_train_questions.json') as f:
        qs = json.load(f)['questions']

    answer2count = {}
    for q in qs:
        answer2count[q['answer']] = answer2count.get(q['answer'], 0) + 1

    answer2prior = {}
    for group, answers_in_group in group2answers.items():
        total_in_group = sum([answer2count.get(answer, 0) for answer in answers_in_group])
        for answer in answers_in_group:
            answer2prior[answer] = 0. if answer2count.get(answer, 0) == 0 else (answer2count[answer] / float(total_in_group))

    nll = np.array([-math.log(answer2prior[q['answer']]) for q in qs]).mean()
    print(f'{template_type}: H(y) = {nll}')
    tt2entropy[template_type] = nll
print(json.dumps(tt2entropy, indent=4))
