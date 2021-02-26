import numpy as np
import os
import pandas as pd
from pprint import pprint

data = pd.read_csv(f'{os.environ["BASE_DIR"]}/gender_words_list.csv')

for gender_subset in [['M', 'F'], ['M'], ['F']]:
    print(f'{gender_subset} SUBSET STATS:\n')
    data_subset = data[np.any([data['gender'] == g for g in gender_subset], axis=0)]
    for k in data_subset.keys():
        if k != 'word':
            v2count = {}
            for v in data_subset[k].values:
                if str(v) != 'nan':
                    v2count[v] = v2count.get(v, 0) + 1
            print(k)
            pprint(v2count)
            print()
