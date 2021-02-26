# Rissanen Data Analysis

Below, we will step through the procedure we used to produce our results.
To see how we train a model on dataset, please skim through the README to see the bash snippet and python script we use to train that model.
To see how we compute minimum description length after training our models, please see `scripts/plot_results_prequential.ipynb`.

Code Overview:
- `film/`: Code for training FiLM (modified from the original [FiLM codebase](https://github.com/ethanjperez/film)). Also includes scripts for 
training FastText models and evaluating MDL.
- `transformers/`: Code for training other, transformer-based models (modified from [HuggingFace 
Transformers](https://github.com/huggingface/transformers)).

## Initial Setup

Clone this repo, then step into the directory and set its path to be `BASE_DIR`, your main working directory:
```bash
git clone https://github.com/ethanjperez/rda.git
cd rda
export BASE_DIR=$PWD
```

You'll need to create a `data` folder (either with `mkdir $BASE_DIR/data` or by symlinking to a location that can hold large files).
Similarly, you'll need to create a `checkpoint` folder for saving model results (either with `mkdir $BASE_DIR/checkpoint` or symlinking).

## Installing dependencies for RDA on HotpotQA/e-SNLI/GLUE/SNLI/ANLI

Install CUDA to train models on GPU. We used CUDA 10.1, but other versions should work as well. You can skip this step if you'd just like to 
reproduce our paper plots from our cached training results (without training models on your own).

Then, setup a Python 3.7+ virtual environment. We [installed Anaconda 3](https://docs.anaconda.com/anaconda/install/) and created a Python 3.7 
conda environment:
```bash
conda create -n rda python=3.7
conda activate rda
```

Next, install PyTorch ([instructions](https://pytorch.org/)). We used PyTorch 1.4 (other versions for both may potentially be compatible), 
using the below command:
```bash
conda install -y pytorch=1.4 torchvision cudatoolkit=10.1 -c pytorch
```
Then, install the remaining dependencies:
```bash
cd $BASE_DIR/rda
pip install -r requirements.txt
python -m spacy download en
pip install --editable .
```

If your GPU supports floating point 16 training, you can train transformer models faster by installing NVIDIA/apex 
([instructions](https://github.com/NVIDIA/apex)), e.g.:
```bash
cd $BASE_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # optional, but helps to install apex in a way that is compatible with many different GPU 
types
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./
```
If you run into installation errors with the above, look through the issues on the [NVIDIA/apex](https://github.com/NVIDIA/apex) repo for your 
installation error, or just skip this part for now.
I was able to fix my installation errors by using an older version of apex, by running `cd $BASE_DIR/apex; git reset --hard de6378f5da` after 
`git clone https://github.com/NVIDIA/apex.git`.

If you'd like to train FastText models (optional), then install FastText like so:
```bash
cd $BASE_DIR
git clone https://github.com/facebookresearch/fastText
cd fastText
pip install .
```
Then, download the pretrained FastText vectors that we used:
```bash
mkdir $BASE_DIR/data/fastText
cd $BASE_DIR/data/fastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip .
unzip crawl-300d-2M.vec.zip
```

To run experiments on CLEVR, please jump to the secion "RDA on CLEVR". Otherwise, continue below.

## Reproducing our plots from cached training results

You can compute MDL and reproduce all of our plots by downloading the 
[results](https://drive.google.com/file/d/1sWcjvOdNg_TEV4jWyY6lX2ToOomiEv9h/view) our of training runs (*skip this step if you'd like to train 
your own models*):
```bash
# Function to google drive from terminal
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 
"https://docs.google.com/uc?export=download&id=$
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Download RDA results
cd $BASE_DIR/checkpoint
gdrive_download 1sWcjvOdNg_TEV4jWyY6lX2ToOomiEv9h rda_results.tar.gz
tar -xvzf rda_results.tar.gz
rmdir rda
mv rda_results rda
```
If you run into issues when downloading from Google Drive with the `gdrive_download` command or when extracting from the downloaded `tar.gz` 
file, just download directly from Google Drive [here](https://drive.google.com/file/d/1sWcjvOdNg_TEV4jWyY6lX2ToOomiEv9h/view).

Then, you can plot our results on CLEVR using:
```bash
cd $BASE_DIR/film
EXP=clevr # Change to plot results for other datasets
python scripts/plot_results.py --exp $EXP
```
Likewise, plot our results for HotpotQA with `EXP=hotpot` and for e-SNLI with `EXP=esnli`.
To plot GLUE/SNLI/ANLI results, set `EXP` to the ablation type (in order of plot appearance in our paper): `pos`, `gender`, `shuffle` (word 
order ablation), `content`, `causal`, `logical`, `gender_frequency_controlled`, `length`.

## RDA on e-SNLI

Download and format (e-)SNLI for model training:
```bash
cd $BASE_DIR/data
mkdir e-SNLI
cd e-SNLI
for SPLIT in "dev" "test" "train_1" "train_2"; do
    wget https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_$SPLIT.csv
done
cd $BASE_DIR/scripts
# Format and save (e-)SNLI training data (with explanations or rationales or neither)
python esnli.format_data.py
```

The above script will save data to different folders `$BASE_DIR/data/$TN` where the data in `TN` is described below:
<table>
<tr>
    <td><b> TN </b></td>
    <td><b> Type of Input Data </b></td>
</tr>
<tr>
    <td> esnli.input-raw </td>
    <td> Normal Input (Premise+Hypothesis) </td>
</tr>
<tr>
    <td> esnli.input-markedonly </td>
    <td> Rationale </td>
</tr>
<tr>
    <td> esnli.input-markedmasked </td>
    <td> Input with Rationale Words Masked </td>
</tr>
<tr>
    <td> esnli.input-markedunmasked </td>
    <td> Input with Non-Rationale Words Masked </td>
</tr>
<tr>
    <td> esnli.input-marked </td>
    <td> Input + Rationale </td>
</tr>
<tr>
    <td> esnli.input-explanation </td>
    <td> Explanation </td>
</tr>
<tr>
    <td> esnli.input-raw,explanation </td>
    <td> Input + Explanation </td>
</tr>
<tr>
    <td> esnli.input-marked,explanation </td>
    <td> Input + Rationale + Explanation </td>
</tr>
</table>

### TODO: Check you can train fasttext models when installing as above
Here is how you can train the FastText classifier on the same data (runs on CPU):
```bash
for PBN in 0 1 2 3 4 5 6 7; do
for TN in "esnli.input-raw" "esnli.input-markedonly" "esnli.input-markedmasked" "esnli.input-markedunmasked" "esnli.input-marked" 
"esnli.input-raw,explanation" "esnli.input-explanation"; do
python $BASE_DIR/scripts/train_fasttext_classifier.py --task_name $TN --prequential_block_no $PBN --autotune_duration 7200 --seeds 12 20 21 22 
23
done
done
```
The above will train five FastText models with 5 different random seeds to evaluate codelengths (loss) for each block of the training data.
It will also tune the softamx temperature of FastText models automatically (via grid search, choosing the best tmeperature based on validation 
loss).
The above will choose hyperparameters using FastText's autotune functionality and then use the chosen hyperparameters from the first random 
seed to train four additional models using different random seeds (for model training and data orderings for online/prequential coding).

Here is how you can train the tranformer-based models that we trained for our e-SNLI experiments (runs on 1 GPU):
```bash
cd $BASE_DIR/transformers
for MN in "distilroberta-base" "distilgpt2" "roberta-base" "gpt2" "facebook/bart-base" "albert-base-v2" "roberta-large" 
"roberta-base.from_scratch" "roberta-large.from_scratch"; do
for TN in "esnli.input-raw" "esnli.input-markedonly" "esnli.input-markedmasked" "esnli.input-markedunmasked" "esnli.input-marked" 
"esnli.input-raw,explanation" "esnli.input-explanation"; do
python online_coding.py --mn $MN --tn $TN --max_pbn 7 --cpus 1  # NB: Increase --cpus if you have more available, for faster data loading and 
preprocessing
done
done
```
The above will train five models with different random seeds for each model class (`roberta-base`, `gpt2`, etc.)  to evaluate codelengths 
(loss) for each block of the training data.
It will run a hyperparameter sweep for a given model class and then use the chosen hyperparameters from the first random seed to train four 
additional models using different random seeds (for model training and data orderings for online/prequential coding).
In practice, you'll probably want to parallelize calls to `online_coding.py` for different model classes and tasks Running the above will take 
a while on a single GPU, so you'll probably want to change the for loop as needed (e.g., just use one random seed or one task) or parallelize 
the above training runs by running each as a separate job on a cluster.
The below command will tune temperatures (parallelized across CPU cores) for all transformer-based models trained above:
```bash
python $BASE_DIR/scripts/tune_temperature_parallelized.py --group esnli
```

Finally, compute MDL using the codelengths obtained from above and save plots to `$BASE_DIR/scripts/plots` with:
```bash
python scripts/plot_results.py --exp esnli
```
*Note 1*: The above script will also tune the softmax temperatures for any models that have not had temperature tuned already via 
`tune_temperature_parallelized.py` (though `tune_temperature_parallelized.py` is parallelized and thus faster).

*Note 2*: If you downloaded our results earlier, you'll want to delete them before running `scripts/plot_results.py`, as the script loads in 
the cached results if they are available. You can delete the cached results like so:
```bash
rm $BASE_DIR/checkpoint/*pbn2seed2hpstr2stats.json $BASE_DIR/checkpoint/*/*pbn2seed2hpstr2stats.json
```

To change the models used in the ensemble (or just use a single model), use the `--model_names` flag, e.g.:
```bash
python scripts/plot_results.py --exp esnli --model_names distilroberta-base distilgpt2  # ensemble these two models
```

To compute the results over different (e.g., fewer) random seeds:
```bash
python scripts/plot_results.py --exp esnli --seeds 12 20 21
```

## RDA on GLUE/ANLI/SNLI

For SNLI, we use the downloaded and formatted data from the e-SNLI section above (saved to `$BASE_DIR/data/esnli.input-raw`).

For GLUE, download its datasets using the GLUE data download script:
```bash
cd $BASE_DIR/data
wget https://raw.githubusercontent.com/nyu-mll/jiant/093c556dc513583770ccad4b3f3daad6f37a7bda/scripts/download_glue_data.py
python download_glue_data.py --data_dir data
rm download_glue_data.py
for TN in "CoLA" "MNLI" "MRPC" "QNLI" "QQP" "RTE" "SNLI" "SST-2" "STS-B" "WNLI"; do
    mv $TN $(echo "$TN" | tr '[:upper:]' '[:lower:]')  # lowercase data directory names
done
```

For ANLI, download the data for each round (1-3) as follows:
```bash
cd $BASE_DIR/data
wget https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip
unzip anli_v1.0.zip
for ROUND in "1" "2" "3"; do
    mv anli_v1.0/R$ROUND anli.round$ROUND
done
# Clean up unnecessary files
rm -r anli_v1.0
rm anli_v1.0.zip
```

For simplicity, we will first show how to run RDA to evaluate the importance of word order on CoLA.
The procedure for evaluating the importance of different features on other GLUE/SNLI/ANLI datasets is the similar (details later).

Create the ablated versions of the original data that we test in our paper (masking different POS words, logical words, etc.):
```bash
cd $BASE_DIR/scripts
TN="cola"
for SPLIT in "train" "dev" "test"; do
python dataset_ablations.py --task $TN --split $SPLIT --shard_no $SHARD_NO --num_shards $NUM_SHARDS
done
```

You'll now have ~20 different ablated versions of the original dataset.
Here, we'll show how to train models on the original vs. word-shuffled text (used for our word order experiments).
Just use a different task name (`TN`) for other dataset ablations, matching the name of the ablation's directory.
For example, to train of CoLA with (1) input nouns masked use `TN=cola.mask_noun` and (2) the same fraction of input words use 
`TN=cola.mask_noun_fraction`.

You can train FastText models on the above data in the same way as for e-SNLI above (just updating `TN`):
```bash
for PBN in 0 1 2 3 4 5 6 7; do
for TN in "cola" "cola.shuffle"; do
python $BASE_DIR/scripts/train_fasttext_classifier.py --task_name $TN --prequential_block_no $PBN --autotune_duration 7200
done
done
```

Similarly, train transformer-based models with:
```bash
cd $BASE_DIR/transformers
for MN in "distilroberta-base" "distilgpt2" "roberta-base" "gpt2" "facebook/bart-base" "albert-base-v2" "roberta-large" 
"roberta-base.from_scratch" "roberta-large.from_scratch"; do
for TN in "cola" "cola.shuffle"; do
python online_coding.py --mn $MN --tn $TN --max_pbn 7 --cpus 1  # NB: Increase --cpus if you have more available, for faster data loading and 
preprocessing
done
done
```
And tune their temperature with:
```bash
python $BASE_DIR/scripts/tune_temperature_parallelized.py --group cola.shuffle
```

Finally, compute MDL using the codelengths obtained from above and save plots to `$BASE_DIR/scripts/plots` with:
```bash
python scripts/plot_results.py --exp shuffle --task_types cola
```
### TODO: Check that above works
### TODO: Add instructions for using `plot_results.py` in general (different seeds, task_types, etc.)
### TODO: Make table showing what each of the ablation names are and what ablations you need to generate a full plot in our paper
### TODO: Elaborate on how to reproduce other results

## RDA on HotpotQA

Download the HotpotQA data, with/without subanswers from different decomposition methods as follows (located on Google drive 
[here](https://drive.google.com/file/d/1QFMJSH5fB_OPH4xoDhvjyAWoUsLQIImw/view)):
```bash
# Function to google drive from terminal
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 
"https://docs.google.com/uc?export=download&id=$
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Download
cd $BASE_DIR/data
gdrive_download 1QFMJSH5fB_OPH4xoDhvjyAWoUsLQIImw hotpot.with_subanswers.tar.gz
tar -xvzf hotpot.with_subanswers.tar.gz
mv hotpot.with_subanswers/* .
rmdir hotpot.with_subanswers
rm hotpot.with_subanswers.tar.gz
```
Again, if you run into issues when downloading from Google Drive with the `gdrive_download` command or when extracting from the downloaded 
`tar.gz` file, just download directly from Google Drive ([here](https://drive.google.com/file/d/1QFMJSH5fB_OPH4xoDhvjyAWoUsLQIImw/view)).
The above will save data to different folders `$BASE_DIR/data/$TN` where the data in `TN` is described below:
<table>
<tr>
    <td><b> TN </b></td>
    <td><b> Decomposition Method </b></td>
</tr>
<tr>
    <td> hotpot.context-long.num_subas-0.shuffled-1 </td>
    <td> No Decomposition </td>
</tr>
<tr>
    <td> hotpot.context-long.subqs-21979200.num_subas-2.shuffled-1 </td>
    <td> Pseudo-Decomposition </td>
</tr>
<tr>
    <td> hotpot.context-long.subqs-20604919.num_subas-2.shuffled-1 </td>
    <td> Seq2Seq </td>
</tr>
<tr>
    <td> hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1 </td>
    <td> ONUS </td>
</tr>
<tr>
    <td> hotpot.context-long.subqs-7.num_subas-2.shuffled-1 </td>
    <td> DLM </td>
</tr>
<tr>
    <td> hotpot.context-long.num_subas-2.shuffled-1 </td>
    <td> Oracle </td>
</tr>
</table>

Then, you can sweep over hyperparameters for pretrained Longformer BASE model on HotpotQA (with subanswers from the above decomposition 
methods) like so:
```bash
cd $BASE_DIR/transformers
NTE=6  # number of training epochs
BS=32  # effective batch size
MN="allenai/longformer-base-4096"  # use "allenai/longformer-base-4096.from_scratch" to train the model from random initialization
MAX_TBS=4  # maximum GPU batch size during training (set for a 48GB GPU, reduce to 2 or 1 as appropriate for your GPU)
for TN in "hotpot.context-long.num_subas-0.shuffled-1" "hotpot.context-long.subqs-21979200.num_subas-2.shuffled-1" 
"hotpot.context-long.subqs-20604919.num_subas-2.shuffled-1" "hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1" 
"hotpot.context-long.subqs-7.num_subas-2.shuffled-1" "hotpot.context-long.num_subas-2.shuffled-1"; do
for PBN in 0 1 2 3 4 5 6 7; do
for SEED in 12; do
for LR in 3e-5 5e-5 1e-4; do
GAS=$((((BS-1)/MAX_TBS)+1))
TBS=$((BS/GAS))
OUTPUT_DIR="checkpoint/rda/tn-$TN.mn-$MN.bs-$BS.lr-$LR.nte-$NTE.seed-$SEED.pbn-$PBN"
CPUS=4  # Number of CPU available on your system (e.g., for data loading/processing)
mkdir -p $OUTPUT_DIR
python examples/question-answering/run_squad.py --model_type longformer --model_name_or_path $MN --data_dir data/$TN --do_train --do_eval 
--dev_file dev1.json --test_file dev2.json --fp16_opt_level 2 --output_dir $OUTPUT_DIR --per_gpu_train_batch_size $TBS 
--per_gpu_eval_batch_size $((2*TBS)) --gradient_accumulation_steps $GAS --learning_rate $LR --max_seq_length 4096 --doc_stride 1024 --seed 
$SEED --max_grad_norm inf --adam_epsilon 1e-6 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs $NTE --threads $CPUS 
--logging_steps 1 --prequential_block_no $PBN --early_stopping --evaluate_loss --overwrite_output_dir
done
done
done
done
```

To train additional random seeds using the best hyperparameters from the above sweep, simply set `LR=0` (e.g., replace `for LR in 3e-5 5e-5 
1e-4; do` with `for LR in 0; done`) and sweep over the `SEED` variable with a for loop (e.g., replace `for SEED in 12; do` with `for SEED in 20 
21 22 23; do`).
Finally, compute MDL and plot/save results:
```bash
python $BASE_DIR/scripts/plot_results.py --exp hotpot
```

As before, you can control which random seeds you use for evaluation with the `--seeds` flag and, e.g., just evaluate on one random seed:
```bash
python $BASE_DIR/scripts/plot_results.py --exp hotpot --seeds 12
```

### Answering Subquestions

#### TODO: Check data setup details below
Here, you can see how to answer subquestions (from [ONUS](https://arxiv.org/abs/2002.09758)) with a pretrained SQuAD (as we did) and add the 
subanswers (paragraph markings) to HotpotQA input paragraphs.
These instructions are useful if you have your own subquestions that you'd like to test.
First, setup the Unsupervised Question Decomposition [repo](https://github.com/facebookresearch/UnsupervisedDecomposition), cloning the repo 
into `$BASE_DIR/`, e.g.:
```bash
git clone
cd UnsupervisedDecomposition
export MAIN_DIR=$PWD
# Setup conda environment and download data and pretrained models according the instructions in 
https://github.com/facebookresearch/UnsupervisedDecomposition
```

Next, activate the virtual environment for that repo and then answer subquestions for HotpotQA training set as follows:
```bash
MODEL_DIR=$MAIN_DIR/XLM/dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223  # ONUS decompositions path. To use other decompositions, 
point this path to a folder containing other decompositions
SPLIT="train"  # use "dev" to answer subquestions for dev set (not necessary for RDA experiments)
ST=0.0
LP=1.0
BEAM=5
SEED=0
MODEL_NO="$(echo $MODEL_DIR | rev | cut -d/ -f1 | rev)"
DATA_FOLDER=data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED

cd $MAIN_DIR/pytorch-transformers

# Convert Sub-Qs to SQUAD format
if [ $SPLIT == "dev" ]; then SUBQS_SPLIT="valid"; else SUBQS_SPLIT="$SPLIT"; fi
python umt_gen_subqs_to_squad_format.py --model_dir $MODEL_DIR --data_folder all --sample_temperature $ST --beam $BEAM --length_penalty $LP 
--seed $SEED --split $SUBQS_SPLIT --new_data_format

for NUM_PARAGRAPHS in 1 3; do
    python examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --train_file $DATA_FOLDER/train.json --predict_file 
$DATA_FOLDER/$SPLIT.json --do_eval --do_lower_case --version_2_with_negative --output_dir 
checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 32 
--learning_rate 1.5e-5 --max_query_length 234 --max_seq_length 512 --doc_stride 50 --num_shards 1 --seed 0 --max_grad_norm inf --adam_epsilon 
1e-6 --adam_beta_2 0.98 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs 2 --write_dir 
$DATA_FOLDER/roberta_predict.np=$NUM_PARAGRAPHS --no_answer_file
done

# Ensemble sub-answer predictions
python ensemble_answers_by_confidence_script.py --seeds_list 1 3 --no_answer_file --split $SPLIT --preds_file1 
data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED/roberta_predict.np={}/nbest_predictions_$SPLIT.json

# Add sub-questions and sub-answers to QA input
python add_umt_subqs_subas_to_q_squad_format_new.py --subqs_dir data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED 
--splits $SPLIT --num_shards 1 --model_dir $MODEL_DIR --sample_temperature $ST --beam $BEAM --length_penalty $LP --seed $SEED --use_easy 
--use_squad "--atype sentence-1-center --subq_model roberta-large-np=1-3 --use_q --use_suba --use_subq"

# Find paragraphs containing sub-answers
python $BASE_DIR/scripts/hotpot.eval_supporting_fact_retrieval.py --model_no $MODEL_NO --num_paragraphs_str "1-3" --split $SPLIT

# Generate a SQuAD-formatted version of the data (and add 0-2 subanswers)
python $BASE_DIR/scripts/hotpot.format_data.py --model_no $MODEL_NO --split $SPLIT
```


### Distilled Language Model Decompositions

Here, we include details on how to train a Distilled Language Model (DLM).
You'll need to include your own language model decompositions for training data, where each input example is formatted on a new line and each 
output is on a new line (with the same line number as the corresponding input).
Put training inputs in a single directory, `DATA_DIR`, containing a file with name `train.source` that contains training inputs and 
`train.target` that contains training outputs/targets.
Likewise, do the same for validation examples (`val.source` and `val.target`) and test examples (`test.source` and `test.target`).

To train DLM, clone my version of HuggingFace Transformers 4.0 (located [here](https://github.com/ethanjperez/distilled-lm)), e.g.:
```bash
cd $BASE_DIR
git clone https://github.com/ethanjperez/distilled-lm.git
cd distilled-lm
```
The repo is HuggingFace Transformers 4.0 with a few bug fixes to get sequence-to-sequence training with T5 to work.
Next, follow that repo's instructions to install it along with the necessary dependencies (e.g., inside a new conda environment).
Now, you can train a Distilled Language Model like so (be sure to update the `DATA_DIR` variable):
```bash
cd $BASE_DIR/distilled-lm/examples/seq2seq
MSL=142  # max seq len
CPUS=10  # number of CPUs available
EB=4  # evaluation number of beams (for beam search
MAX_TBS=16  # max training batch size that will fit on GPU (currently set for a 48GB GPU, decrease for GPUs with less memory)
BS=64  # effective training batch size (after gradient accumulation)
GAS=$((((BS-1)/MAX_TBS)+1))  # gradient accumulation steps
TBS=$((BS/GAS))  # GPU batch size during training (per forward pass)
OUTPUT_DIR="$BASE_DIR/checkpoint/dlm"
mkdir -p $OUTPUT_DIR
DATA_DIR="/path/to/decompositions"  # TODO: Set the data directory containing decompositions you'd like to train on
# Train DLM model
python finetune.py --data_dir=$DATA_DIR --learning_rate=1e-4 --train_batch_size=$TBS --eval_batch_size=$((2*TBS/EB)) --output_dir=$OUTPUT_DIR 
--max_source_length=$MSL --max_target_length=$MSL --n_val=-1 --do_train --do_predict --model_name_or_path "t5-3b" --adafactor --gpus 1 
--gradient_accumulation_steps $GAS --num_workers $CPUS --val_check_interval=1.0 --warmup_steps 0 --max_grad_norm inf --task translation 
--val_metric bleu --early_stopping_patience 1 --eval_beams $EB --label_smoothing 0.1 --num_train_epochs 3
```

Then, set the `QUESTIONS_PATH` variable to a file containing (line-by-line) the questions you'd like to decompose.
Also set the `GENERATION_OUTPUT_PATH` variable to the filepath where you'd like to save question decompositions.
You can then generate decompositions with the trained DLM as follows:
```bash
QUESTIONS_PATH="/path/to/questions/to/be/decomposed.txt"
GENERATION_OUTPUT_PATH="/path/to/decomposed/questions.txt"
python run_eval.py "$OUTPUT_DIR/best_tfmr" "$QUESTIONS_PATH" "$GENERATION_OUTPUT_PATH" --reference_path "$QUESTIONS_PATH" --task translation 
--bs 8 --num_beams 4 --length_penalty 0.6 --early_stopping true
```

We then apply some postprocessing (to remove excess punctuation) and lowercasing (to match how the ) to generated question decompositions:
```bash
python $BASE_DIR/scripts/hotpot.postprocess_dlm_decompositions.py --gen_path $GENERATION_OUTPUT_PATH
```

Then, you can follow the instructions from earlier/above to answer subquestions and add subanswers to input paragraphs.


## RDA on CLEVR

Install CUDA to run on GPU. We used CUDA 9.2 with GCC 6.3.0, but other versions should work as well.
Then, setup a Python 3.7+ virtual environment. We [installed Anaconda 3](https://docs.anaconda.com/anaconda/install/) and created a Python 3.7 
conda environment:
```bash
conda create -n film python=3.7
conda activate film
# Install PyTorch (Instructions here: https://pytorch.org/). We used the below command:
conda install pytorch=0.4.1 torchvision=0.2.1 cuda92 -c pytorch
# Install film repo dependencies
pip install numpy Pillow scipy==1.1.0 h5py tqdm ipdb termcolor matplotlib
# Install RDA script dependencies
pip install pandas
cd $BASE_DIR/transformers
pip install --editable .
```

Next, follow the data download and preprocessing instructions [here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md) to 
save the data into `data/CLEVR_v1.0`.
This process takes some time, because there are a large number of image files to unzip and extract features for using a pretrained ImageNet 
model.

Next, find the three subsets of CLEVR questions that we use in our paper and append answers to subquestions to the CLEVR question:
```bash
cd $BASE_DIR/data
mkdir CLEVR_1.0_templates
cd CLEVR_1.0_templates
for QTYPE in "comparison" "compare_integer" "same_relate"; do
    wget https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/master/question_generation/CLEVR_1.0_templates/$QTYPE.json
done
python $BASE_DIR/clevr.format_data.py  # Takes ~5 minutes
```

Then, preprocess the question files generated above:
```bash
cd $BASE_DIR/film/
for num_sas in 0 1 2; do
for QTYPE in "comparison" "compare_integer" "same_relate"; do
questions_dirname="questions.$QTYPE.num_sas-$num_sas"
if [[ "$questions_dirname" != "questions.same_relate.num_sas-2" ]]; then
mkdir -p "data/CLEVR_v1.0/$questions_dirname"
sh scripts/preprocess.sh $questions_dirname
fi
done
done
```

As shown below, you can then train models on CLEVR (all 3 question families we tested, with various numbers of subanswers, 5 random seeds):
```bash
cd $BASE_DIR/film
MN="FiLM"
BS=64
LR="3e-4"
NTE=20
for SEED in 12 20 21 22 23; do
for TN in "clevr.comparison.num_sas-$SAS" "clevr.compare_integer.num_sas-$SAS" "clevr.same_relate.num_sas-$SAS"; do
for SAS in 0 1 2; do
for PBN in 0 1 2 3 4 5 6 7; do
if [[ "$TN" != "clevr.same_relate.num_sas-2" ]]; then
OUTPUT_DIR="exp/rda/tn-$TN.mn-$MN.bs-$BS.lr-$LR.nte-$NTE.seed-$SEED.pbn-$PBN"
mkdir -p $OUTPUT_DIR
sh scripts/train/film_prequential.sh $TN $MN $BS $LR $NTE $SEED $PBN
fi
done
done
done
done
```

Each call to `scripts/train/num_train_samples.modified_questions.prequential.sh` trains a single FiLM model for a particular random seed 
(`SEED`), task (with name `TN`, i.e., the question type), number of subanswers (`SAS`), and prequential block number (`PBN` indicates training 
on all blocks of data up to and including block number `PBN`).
Running the above will take a while on a single GPU, so you'll probably want to change the for loop as needed (e.g., just use one random seed 
or one task) or parallelize the above training runs by running each as a separate job on a cluster.

Finally, compute MDL and save plots to `$BASE_DIR/scripts/plots` by deactivating your current environment, activating the `rda` environment for 
non-CLEVR experiments, and then running:
```bash
python scripts/plot_results.py --exp clevr
```

If you just want to compute MDL for one question type (`TN` above), just run:
```bash
TN="clevr-compare_integer"  # in {clevr-compare_integer,clevr-comparison,clevr-same_relate}, or add multiple with one space separating each
python scripts/plot_results.py --exp clevr --task_types $TN
```

*Note*: Please install any missing packages you encounter along the way above using `pip install [dependency]` (e.g., if you encounter 
`ModuleNotFoundError: No module named [dependency]`).

## Other Scripts

To compute `H(y)` (the label entropy baseline) for CLEVR, run:
```bash
python $BASE_DIR/scripts/clevr.compute_entropy.py
```
and for GLUE/(e)SNLI/ANLI, run:
```bash
python $BASE_DIR/scripts/glue.compute_entropy.py
```

## Bibtex Citation

```bash
@article{perez2021rissanen,
  author = {Ethan Perez and Douwe Kiela and Kyunghyun Cho},
  title = {Rissanen Data Analysis: Examining Dataset Characteristics via Description Length},
  journal={arXiv},
  year = {2021}
}
```

