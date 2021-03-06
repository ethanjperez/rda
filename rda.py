import argparse
import math
import os
import random
import sys
import transformers.run_glue as train_model  # transformers only required for demo purposes - change import to load a training function for other models


def load_data(data_dir):
    """
    Here, we load a dataset from GLUE (MRPC's training set) using the HuggingFace "datasets" library.
    Replace the code below to load a different dataset (e.g., from data_dir) into a list of data instances (the instances can have any data type).
    """
    from datasets import load_dataset
    dataset = list(load_dataset("glue", "mrpc", split='train'))
    # Optionally, you can augment or ablate the input data here
    return dataset


def save_data(examples, save_file):
    """
    Save list of data instances to a file that can be read by the model training function.
    Below, we save instances in a way that is compatible with loading data via HuggingFace datasets.
    Replace the code below if you'd like to save data to a different format (as required by your model training function).
    """
    import json
    with open(save_file, 'w') as f:
        f.writelines('\n'.join([json.dumps(ex) for ex in examples]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_args", default='', type=str, help="Training arguments to pass to the model training function,"
                        "e.g., --learning rate 3e-5 --train_file TRAIN_FILE --validation_file VALIDATION_FILE --test_file TEST_FILE'."
                        "Must contain 'TRAIN_FILE', 'VALIDATION_FILE', and 'TEST_FILE', as these strings will be replaced by the filepaths"
                        "for train/valid/test data when training models on different data blocks.")
    parser.add_argument("--data_dir", default='data', type=str, help="The directory in which RDA train/val/test data will be saved."
                        "If your load_data function requires a data path, this directory should also contain the original data you would like to analyze.")
    parser.add_argument("--data_file_ext", default='json', type=str, help="File extension for saved RDA train/val/test data.")
    parser.add_argument("--num_blocks", default=9, type=int, help="The number of blocks N sent when calculating rda code length"
                        "(trains N-1 models, as the first block is sent with a uniform prior).")
    parser.add_argument("--min_num_train_samples", default=64, type=int, help="The minimum number of examples to train with.")
    parser.add_argument("--max_num_train_samples", default=float('inf'), type=int, help="The maximum number of examples to use. 0 for all examples.")
    parser.add_argument("--val_frac", default=0.1, type=float, help="The fraction of training examples to split off for validation.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed to use for online coding (e.g., for randomly ordering examples).")
    parser.add_argument("--label_range", required=True, type=float, help="For tasks with a discrete output space (e.g., classification or span prediction),"
                        "use the number of possible output classes. For regression, use the size of the interval over which outputs can range, e.g.,"
                        "3.5 if the range is [1., 4.5]. This value is used to calculate the codelength for sending the first block using the uniform prior."
                        "For regression, the size of the interval over which outputs can range, e.g., 3.5 if the range is [1., 4.5]")
    parser.add_argument("--mse", default=False, action="store_true", help="Use this flag if returning mean-squared error (MSE)"
                        "instead of negative log-likelihood (we'll convert MSE values to NLL). Please do not divide MSE values by 2,"
                        "i.e., just return the average value of (y' - y) ^ 2, where y is the true label and y' is the predicted label.")
    args = parser.parse_args()

    # Check arguments
    assert args.num_blocks >= 1, '--num_blocks must be >= 1'
    assert args.min_num_train_samples >= 1, '--min_num_train_samples must be >= 1'
    assert args.max_num_train_samples >= args.min_num_train_samples, '--max_num_train_samples must be >= --min_num_train_samples'
    assert 0 <= args.val_frac < 1, '--val_frac must be >= 0 and < 1'
    assert args.label_range > 0, '--label_range must be > 0'

    # Load and shuffle data
    dataset = load_data(args.data_dir)
    rng = random.Random(args.seed)
    rng.shuffle(dataset)

    # Compute data block sizes
    log_block_size_increment = (math.log(min(len(dataset), args.max_num_train_samples)) - math.log(args.min_num_train_samples)) / (args.num_blocks - 1)
    block_start_idxs = [0] + [int(round(math.exp(math.log(args.min_num_train_samples) + (block * log_block_size_increment)))) for block in range(args.num_blocks)]
    block_sizes = [(block_start - block_end) for block_start, block_end in zip(block_start_idxs[1:], block_start_idxs[:-1])]

    # Collect negative log-likelihoods (in nats, i.e., base e) for sending each block below
    nlls = []
    # Add the NLL for sending the first block with the uniform prior
    nlls.append(-math.log(1. / float(args.label_range)))

    # Create train/val/test splits for sending each data block after the first
    for send_block in range(1, args.num_blocks):
        train_val_dataset = dataset[:block_start_idxs[send_block]]
        rng.shuffle(train_val_dataset)
        val_size = int(round(args.val_frac * len(train_val_dataset)))
        block_datasets = {
            'train': train_val_dataset[val_size:],
            'validation': train_val_dataset[:val_size],
            'test': dataset[block_start_idxs[send_block]: block_start_idxs[send_block + 1]],
        }

        # Save train/val/test data and add data paths to model training arguments
        block_data_dir = os.path.join(args.data_dir, 'send_block_' + str(send_block))
        os.makedirs(block_data_dir, exist_ok=True)
        block_training_args = args.training_args
        for split, block_dataset in block_datasets.items():
            block_split_filepath = os.path.join(block_data_dir, split + '.' + args.data_file_ext)
            print('Saving data to:', block_split_filepath)
            save_data(block_dataset, block_split_filepath)
            assert (split.upper() + '_FILE') in args.training_args, 'Expected ' + split.upper() + '_FILE in args.training_args'
            block_training_args = block_training_args.replace(split.upper() + '_FILE', block_split_filepath)

        sys.argv = [train_model.__file__] + block_training_args.split()  # Set command line args for model training
        test_loss = train_model.main()  # Call main function to train model with above args, to get test NLL on this block
        if args.mse:
            std_dev = 1.  # Treat all regression/MSE predictions as a mean with this std. dev. We use 1 as a default, but other values may work better, e.g., if chosen on dev
            nlls.append((test_loss / (2. * (std_dev ** 2))) + math.log(std_dev * math.sqrt(2 * math.pi)))  # Convert MSE loss to Mean NLL
        else:
            nlls.append(test_loss)  # loss for classification is NLL

    # Compute MDL
    codelengths = [nll / math.log(2) for nll in nlls]
    print('Per-sample codelengths (in bits) for different blocks:\n\t', codelengths)
    mdl = sum(block_size * per_sample_codelength for block_size, per_sample_codelength in zip(block_sizes, codelengths))
    print('MDL:', mdl, 'bits')
