import argparse
import os
import re


multispace = re.compile('\s+')
multiq = re.compile('\?+')
multiexclamation = re.compile('\!+')
multiperiod = re.compile('\.+')
multispaceq = re.compile('(\s\?)+')


def process(text):
    text = text.replace(' ?"', '"?')
    text = text.replace('?"', '"?')
    text = text.replace('?', ' ? ')
    text = multispace.sub(' ', text)
    text = multiexclamation.sub('!', text)
    text = multiperiod.sub('.', text)
    text = multiq.sub('?', text)
    text = multispaceq.sub(' ?', text)
    text = text.replace(' s ', ' ')

    for pattern in ['! ?', '!?', '?!', '? !', '? !', '? .', '. ?', '?.', '.?', '? ?', '??']:
        text = text.replace(pattern, '?')
    return text.strip()


def process_until_convergence(text):
    text = text.replace('\t', ' ')

    proc_text = process(text)
    while proc_text != text:
        text = proc_text
        proc_text = process(proc_text)
    return proc_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True, help="Absolute path containing decompositions.")
    args = parser.parse_args()

    save_dir, filename = args.gen_path.rsplit('/', 1)
    save_dir += '.clean'
    os.makedirs(save_dir, exist_ok=True)
    save_filepath = f'{save_dir}/{filename}'

    with open(args.gen_path) as f:
        data = [line.strip() for line in f]

    data_proc = [process_until_convergence(d).lower() for d in data]
    num_cleaned = sum(d != dp for d, dp in zip(data, data_proc))
    print(f'Decompositions Cleaned: {num_cleaned} / {len(data)} ({round(100. * num_cleaned / len(data), 2)}%)')

    with open(save_filepath, 'w') as f:
        f.writelines('\n'.join(data_proc) + '\n')
    print('Done! Saved to:', save_filepath)
