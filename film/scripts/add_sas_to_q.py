from copy import deepcopy
from datetime import datetime
import json
import os
import re
from tqdm.auto import tqdm


DATA_DIR = os.path.join(os.environ['BASE_DIR'], 'data')
splits = ['val', 'train']


def execute_program(program, scene):
    for o in range(len(scene['objects'])):
        scene['objects'][o]['object_index'] = o

    outs = []
    for f in program:
        inputs = [outs[input_idx] for input_idx in f['inputs']]
        outs.append(apply_function(f, inputs, scene))
    return [out for out in outs]


def apply_function(f, inputs, s):
    """
    Interpreter for CLEVR proram function calls.
    Returns a string or integer.
    """

    if f['function'] == 'scene':
        assert len(inputs) == 0
        assert len(f['value_inputs']) == 0

        return s['objects']

    elif f['function'].startswith('filter'):
        prop = f['function'].split('_', 1)[1]
        assert prop in {'color', 'material', 'shape', 'size'}
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 1

        return [obj for obj in inputs[0] if obj[prop] == f['value_inputs'][0]]

    elif f['function'] == 'unique':
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 0

        assert len(inputs[0]) == 1, 'Expected a single, unique object in object list'
        return inputs[0][0]

    elif f['function'].startswith('query'):
        prop = f['function'].split('_', 1)[1]
        assert prop in {'color', 'material', 'shape', 'size'}
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 0

        return inputs[0][prop]

    elif f['function'] == 'relate':
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 1

        direction = f['value_inputs'][0]
        out_object_indexes = s['relationships'][direction][inputs[0]['object_index']]
        return [obj for o, obj in enumerate(s['objects']) if o in out_object_indexes]

    elif f['function'].startswith('same'):
        prop = f['function'].split('_', 1)[1]
        assert prop in {'color', 'material', 'shape', 'size'}
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 0

        inp_obj = inputs[0]
        return [obj for obj in s['objects']
                if ((obj[prop] == inp_obj[prop]) and (obj['object_index'] != inp_obj['object_index']))]

    elif f['function'] == 'exist':
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 0

        return 'yes' if len(inputs[0]) >= 1 else 'no'

    elif f['function'] == 'count':
        assert len(inputs) == 1
        assert len(f['value_inputs']) == 0

        return len(inputs[0])

    elif f['function'] == 'union':
        assert len(inputs) == 2
        assert len(f['value_inputs']) == 0

        obj_list_indexes = set([obj['object_index'] for obj in inputs[0] + inputs[1]])
        return [obj for obj in s['objects'] if obj['object_index'] in obj_list_indexes]

    elif f['function'] == 'intersect':
        assert len(inputs) == 2
        assert len(f['value_inputs']) == 0

        obj_list_indexes1 = set(obj['object_index'] for obj in inputs[0])
        obj_list_indexes2 = set(obj['object_index'] for obj in inputs[1])
        return [obj for obj in s['objects']
                if ((obj['object_index'] in obj_list_indexes1) and (obj['object_index'] in obj_list_indexes2))]

    elif f['function'].startswith('equal'):
        prop = f['function'].split('_', 1)[1]
        assert prop in {'color', 'material', 'shape', 'size', 'integer'}
        assert len(inputs) == 2
        assert len(f['value_inputs']) == 0

        return 'yes' if inputs[0] == inputs[1] else 'no'

    elif f['function'] == 'greater_than':
        assert len(inputs) == 2
        assert len(f['value_inputs']) == 0
        assert isinstance(inputs[0], int)
        assert isinstance(inputs[1], int)

        return 'yes' if inputs[0] > inputs[1] else 'no'

    elif f['function'] == 'less_than':
        assert len(inputs) == 2
        assert len(f['value_inputs']) == 0
        assert isinstance(inputs[0], int)
        assert isinstance(inputs[1], int)

        return 'yes' if inputs[0] < inputs[1] else 'no'

    else:
        raise NotImplementedError(f'No interpreter implementation for function {f["function"]}')


qs = {}
for split in splits:
    print(f'Loading {split} questions...')
    with open(f'{DATA_DIR}/CLEVR_v1.0/questions/CLEVR_{split}_questions.json') as f:
        qs[split] = json.load(f)['questions']

scenes = {}
for split in splits:
    print(f'Loading {split} scenes...')
    with open(f'{DATA_DIR}/CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json') as f:
        scenes[split] = json.load(f)['scenes']

template_type2subanswer_function = {'comparison': 'query', 'compare_integer': 'count'}
template_type2num_subanswers = {'comparison': 2, 'compare_integer': 2, 'same_relate': 1}

for template_type in ['comparison', 'compare_integer', 'same_relate']:
    templates = json.load(open(f'{DATA_DIR}/CLEVR_1.0_templates/{template_type}.json'))
    text_templates = []
    for template in templates:
        for text in template['text']:
            for pattern in ['2>', '3>', '4>', '5>', '6>', '7>', '8>', '9>']:
                text = text.replace(pattern, '>')
            text = text.replace('<Z> <C> <M> <S>', '.*')
            text_templates.append(text)

    for num_subanswers_to_add in range(template_type2num_subanswers[template_type] + 1):  # Using [# of sub-answers + 1] will include answer
        data_with_subanswers_dir = f'{DATA_DIR}/CLEVR_v1.0/questions.{template_type}.num_sas-{num_subanswers_to_add}'
        os.makedirs(data_with_subanswers_dir, exist_ok=True)

        for split in splits:
            data_with_subanswers = {
                'info': {
                    'split': split,
                    'license': 'Creative Commons Attribution (CC BY 4.0)',
                    'version': template_type,
                    'date': datetime.today().strftime('%m/%d/%y'),
                },
                'questions': [],
            }

            for q in tqdm(qs[split]):
                # Skip questions that don't match a pattern
                matches_pattern = False
                for text_template in text_templates:
                    if re.match(text_template, q['question']):
                        matches_pattern = True
                        break
                if not matches_pattern:
                    continue

                # Execute program on scene
                outs = execute_program(q['program'], scenes[split][q['image_index']])
                assert q['answer'] == str(outs[-1]), f'Incorrect answer: {q["answer"]} != {outs[-1]}'

                # Add subanswer(s) to question
                q_copy = deepcopy(q)
                if template_type == 'same_relate':
                    subanswers = [str(outs[f_idx-1][f['function'].split('_', 1)[1]])
                                  for f_idx, f in enumerate(q['program'])
                                  if f['function'].startswith('same')]
                else:
                    subanswers = [str(outs[f_idx]) for f_idx, f in enumerate(q['program'])
                                  if f['function'].startswith(template_type2subanswer_function[template_type])]
                assert len(subanswers) == template_type2num_subanswers[template_type], f'Expected {template_type2num_subanswers[template_type]} subanswers but found {len(subanswers)}'
                for sa_no in range(num_subanswers_to_add):
                    q_copy['question'] += ' ' + str((subanswers + [q['answer']])[sa_no])

                data_with_subanswers['questions'].append(q_copy)

            data_with_subanswers_file = f'{data_with_subanswers_dir}/CLEVR_{split}_questions.json'
            with open(data_with_subanswers_file, 'w') as f:
                json.dump(data_with_subanswers, f, indent=2)
            print(f'Saved {len(data_with_subanswers["questions"])} examples to {data_with_subanswers_file}')
