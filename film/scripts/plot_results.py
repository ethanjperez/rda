import argparse
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from transformers.loss_data_utils import load_losses, mn2name, tn2name, group2datasethps, plot_mdls, plot_loss_data_curve_prequential, plot_wordtype_importances, get_mn2tn2lossmatrix, get_ablation2mdl, get_ablation2mdlincrease, plot_loss_reduction_by_loss_with_error, cm, glue_nli_tasks, PLOTS_DIR, save_and_show_plot

matplotlib.use("agg")  # no display to GUI (don't use this for e.g. jupyter notebook)

# Add a new config below to look at results for different combinations of tasks, ablations, etc.
exp2config = {
    'clevr': {
        'task_types': ('clevr-compare_integer', 'clevr-comparison', 'clevr-same_relate'),
        'analysis_type': '',
        'ensemble': False,
    },
    'hotpot': {
        'task_types': ('hotpot',),
        'analysis_type': '',
        'ensemble': False,
    },
    'esnli': {
        'task_types': ('esnli',),
        'analysis_type': '',
        'ensemble': True,
    },
    'pos': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'pos',
        'ensemble': True,
    },
    'gender': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'gender',
        'ensemble': True,
    },
    'shuffle': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'shuffle',
        'ensemble': True,
    },
    'content': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'content',
        'ensemble': True,
    },
    'logical': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'logical',
        'ensemble': True,
    },
    'causal': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'causal',
        'ensemble': True,
    },
    'gender_frequency_controlled': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'gender',
        'ensemble': True,
    },
    'length': {
        'task_types': glue_nli_tasks,
        'analysis_type': 'length',
        'ensemble': True,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, choices=list(sorted(exp2config.keys())), required=True, help="Set of experiments to load and compute description lengths for.")
parser.add_argument("--task_types", type=str, default=None, nargs='+', help="Task types to load, e.g., 'cola sst-2' (if not using default).")
parser.add_argument("--ensemble", type=int, default=None, choices=[0, 1], help="Whether or not to use ensemble (omit to use default).")
parser.add_argument("--seeds", type=str, default=None, nargs='+', help="Seeds to load, e.g., '12 20 21 22 23' (if not using default).")
parser.add_argument("--model_names", type=str, default=None, nargs='+', help="Models to load, e.g., 'roberta-base roberta-large' (if not using default).")
args = parser.parse_args()

# Load configuration results
if args.exp == 'gender_frequency_controlled':
    exp = 'gender'
    gender_frequency_controlled = True
else:
    exp = args.exp
    gender_frequency_controlled = False
config = exp2config[exp]
task_types = args.task_types if args.task_types is not None else config['task_types']
analysis_type = config['analysis_type']
ensemble = bool(args.ensemble) if args.ensemble is not None else config['ensemble']
do_tune_temperature = ensemble
cache_mn2tn2losses = ensemble
seeds = [int(seed) for seed in args.seeds] if args.seeds is not None else [12, 20, 21, 22, 23]

savefig = True
os.makedirs(PLOTS_DIR, exist_ok=True)
mn2tn2losses = load_losses(task_types, analysis_type, do_tune_temperature, cache_mn2tn2losses, seeds, args.model_names)

# Ensemble by taking the min loss at each block ('min-loss')
if ensemble:
    all_tns = set([])
    for mn, tn2losses in mn2tn2losses.items():
        for tn, losses in tn2losses.items():
            all_tns.add(tn)
    all_tns = list(all_tns)

    mn2tn2losses['min-loss'] = {}
    for tn in all_tns:
        mn_losses = np.array([tn2losses[tn] for mn, tn2losses in mn2tn2losses.items() if mn != 'min-loss'])
        assert len(mn_losses.shape) == 3, f'Unexpected mn_losses shape ({len(mn_losses.shape)}). Do all losses array have the same shape? Are you missing runs?'
        mn2tn2losses['min-loss'][tn] = np.min(mn_losses, axis=0).tolist()

# Plot codelengths and MDLs from online code training
num_trained_blocks = 8
mn2tn2wordtype2importance = {}
mn2tn2wordtype2importancestderr = {}
mn2tn2ablation2mdlincrease = {}
mn2tn2ablation2mdlincreasestderr = {}
mn2tn2ablation2mdl = {}
for tt_no, task_type in enumerate(task_types):
    group = task_type
    if len(analysis_type) > 0:
        group += f'.{analysis_type}'
    tns = group2datasethps[group]['tn']
    mns = args.model_names if args.model_names is not None else deepcopy(group2datasethps[group]['mn'])

    regression = task_type.lower().startswith('sts-b')
    plot_mns = mns
    extra_bits = 0
    if ensemble:
        # Add bits for sending which model in ensemble was used
        extra_bits = num_trained_blocks * np.log2(len(mns))
        ensemble_mns = ['min-loss']
        for mn in ensemble_mns:
            if mn not in mns:
                mns.append(mn)
        # Only show results for ensemble model
        plot_mns = ['min-loss']

    mn2tn2lossmatrix = get_mn2tn2lossmatrix(mns, tns, task_type, mn2tn2losses)

    legend_params = {'bbox_to_anchor': (1, 1)}
    if exp == 'clevr':
        if task_type == 'clevr-same_relate':
            legend_params = {'loc': 'upper right', 'fontsize': 13}
        else:
            legend_params = {'bbox_to_anchor': (1, 1), 'loc': 'upper right', 'fontsize': 13}
    elif exp == 'esnli':
        legend_params = {'bbox_to_anchor': (-.015, -.015), 'loc': 'lower left', 'fontsize': 11}
    elif exp == 'hotpot':
        legend_params = {'bbox_to_anchor': (-.015, -.015), 'loc': 'lower left', 'fontsize': 13}

    for mn in plot_mns:
        plot_loss_data_curve_prequential(tns, mn, group, mn2tn2lossmatrix, savefig,
                                         legend_params=legend_params,
                                         log_log_scale=(exp in {'hotpot', 'esnli'}),
                                         use_mn_in_ylabel=(exp in {'hotpot'}),
                                         short_ylabel=(exp == 'esnli'),
                                         override_cm=(plt.cm.turbo if exp == 'esnli' else None),
                                         crop_first_block=(exp in {'hotpot'}))
        tn2mdls = plot_mdls(tns, mn, group, mn2tn2lossmatrix, savefig,
                            figsize=(3, 4) if exp in {'clevr', 'hotpot', 'esnli'} else None,
                            broken_axis=exp == 'hotpot',
                            override_cm=(plt.cm.turbo if exp == 'esnli' else None),
                            xlabels_rotation=45 if exp in {'hotpot'} else (50 if exp in {'esnli'} else 30),
                            extra_bits=extra_bits)
        if exp == 'hotpot':
            plot_loss_reduction_by_loss_with_error(tns, mn, group, mn2tn2lossmatrix[mn], savefig)
        if exp in {'pos', 'gender', 'content', 'logical', 'causal', 'male', 'female'}:
            if mn not in mn2tn2wordtype2importance:
                mn2tn2wordtype2importance[mn] = {}
            if mn not in mn2tn2wordtype2importancestderr:
                mn2tn2wordtype2importancestderr[mn] = {}
            mn2tn2wordtype2importance[mn][task_type], mn2tn2wordtype2importancestderr[mn][task_type] = plot_wordtype_importances(tns, mn, mn2tn2lossmatrix, group, savefig, None if gender_frequency_controlled else exp)

            if mn not in mn2tn2ablation2mdlincrease:
                mn2tn2ablation2mdlincrease[mn] = {}
            if mn not in mn2tn2ablation2mdlincreasestderr:
                mn2tn2ablation2mdlincreasestderr[mn] = {}
            mn2tn2ablation2mdlincrease[mn][task_type], mn2tn2ablation2mdlincreasestderr[mn][task_type] = get_ablation2mdlincrease(tns, mn, mn2tn2lossmatrix)
        if analysis_type in {'shuffle', 'length', 'mask_all'}:
            if mn not in mn2tn2ablation2mdl:
                mn2tn2ablation2mdl[mn] = {}
            mn2tn2ablation2mdl[mn][task_type] = get_ablation2mdl(tns, mn, mn2tn2lossmatrix, tn_pattern=f'.{analysis_type}')

# Plot length and shuffle MDL results compared to p(y) and original input
if analysis_type in {'length', 'shuffle'}:

    ablation_name = 'Shuffled' if analysis_type == 'shuffle' else 'Length Only'

    original_means = []
    original_stderrs = []
    shuffled_means = []
    shuffled_stderrs = []
    labels = []
    for tn, ablation2mdl in mn2tn2ablation2mdl['min-loss'].items():
        labels.append(tn2name(tn))
        mean, stderr = ablation2mdl['Original']
        original_means.append(mean)
        original_stderrs.append(stderr)
        mean, stderr = ablation2mdl[ablation_name]
        shuffled_means.append(mean)
        shuffled_stderrs.append(stderr)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, shuffled_means, width, yerr=shuffled_stderrs, label=ablation_name, color=cm(0.5))
    ax.bar(x + width/2, original_means, width, yerr=original_stderrs, label='Original', color=cm(0))

    ax.set_ylabel('MDL normalized by $\mathcal{H}(y)$', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18, rotation=40, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend()
    ax.set_ylim(top=1)

    fig.tight_layout()
    save_and_show_plot(plt, f'shuffle_plot.exp-{exp}.mn-{mn2name[mn]}', savefig)

# Plot MDL-difference bar plots
if exp in {'gender', 'content', 'logical', 'causal', 'male', 'female'} and not gender_frequency_controlled:
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    labels = []
    for tn_no, (tn, wordtype2importance) in enumerate(mn2tn2wordtype2importance[mn].items()):
        for wordtype, importance in wordtype2importance.items():
            yerr=mn2tn2wordtype2importancestderr[mn][tn][wordtype]
            if exp == 'content':
                if importance >= 250:
                    print(f'{tn2name(tn)}: {int(round(importance))}+/-{int(round(yerr))}')
                    importance = 250
                    yerr = 0
            plt.bar(tn_no, importance, yerr=yerr, label=tn2name(tn), color=cm(tn_no / len(mn2tn2wordtype2importance[mn])))
            labels.append(tn2name(tn))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
    if exp in {'gender'}:
        plt.title('Gender Bias Evaluation', fontsize=16)
        plt.ylabel('$MDL_{-Male} - MDL_{-Female}$', fontsize=16)
    else:
        plt.ylabel('$MDL_{-' + exp.capitalize() + '} - MDL_{-Random}$', fontsize=16)
    if exp == 'content':
        ax.set_ylim(top=250)
    save_and_show_plot(plt, f'exp-{exp}.mn-{mn2name[mn]}.summary', savefig)

# Plot word frequency -controlled gender experiment results (in Appendix)
if gender_frequency_controlled:
    width = 0.35
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    labels = []
    for tn_no, (tn, wordtype2importance) in enumerate(mn2tn2wordtype2importance[mn].items()):
        wordtypes = list(sorted(wordtype2importance.keys()))
        for wordtype_no, wordtype in enumerate(wordtypes):
            importance = wordtype2importance[wordtype]
            yerr = mn2tn2wordtype2importancestderr[mn][tn][wordtype]
            xloc = (tn_no - (width / 2.)) if wordtype_no == 0 else (tn_no + (width / 2.))
            plt.bar(xloc, importance, width=width, yerr=yerr, label=wordtype.split(' ')[0] if tn_no == 0 else None, color=cm((0.5 + wordtype_no) / len(wordtype2importance.items())))
            if wordtype_no == 0:
                labels.append(tn2name(tn))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
    if exp in {'gender'}:
        plt.ylabel('$MDL_{-Gender} - MDL_{-Random}$', fontsize=16)
    else:
        plt.ylabel('$MDL_{-' + exp.capitalize() + '} - MDL_{-Random}$', fontsize=16)
    if exp == 'content':
        ax.set_ylim(top=250)
    plt.legend(loc='lower left', title='Gender')
    plt.tight_layout()
    save_and_show_plot(plt, f'exp-gender_frequency_controlled.mn-{mn2name[mn]}.summary', savefig)

if exp in {'gender', 'content', 'logical', 'causal'}:
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    labels = []
    for tn_no, (tn, wordtype2importance) in enumerate(mn2tn2wordtype2importance[mn].items()):
        for wordtype, importance in wordtype2importance.items():
            plt.bar(tn_no, importance, yerr=mn2tn2wordtype2importancestderr[mn][tn][wordtype], label=tn2name(tn), color=cm(tn_no / len(mn2tn2wordtype2importance[mn])))
            labels.append(tn2name(tn))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if exp in {'gender'}:
        plt.title('Gender Bias Evaluation')
        plt.ylabel('$MDL_{-Male} - MDL_{-Female}$', fontsize=16)
    else:
        plt.ylabel('$MDL_{-' + exp.capitalize() + '} - MDL_{-Random}$', fontsize=16)
    save_and_show_plot(plt, f'exp-{exp}.mn-{mn2name[mn]}.summary', savefig)

# Plot heatmap summary
if exp == 'pos':
    mn2tn2wordtype2importancesig = {}
    for mn, tn2wordtype2importance in mn2tn2wordtype2importance.items():
        mn2tn2wordtype2importancesig[mn] = {}
        for tn, wordtype2importance in tn2wordtype2importance.items():
            mn2tn2wordtype2importancesig[mn][tn] = {}
            for wordtype, importance in wordtype2importance.items():
                stderr = mn2tn2wordtype2importancestderr[mn][tn][wordtype]
                upper = importance + stderr
                lower = importance - stderr
                mn2tn2wordtype2importancesig[mn][tn][wordtype] = (upper * lower) > 0
    tn2wordtype2importancesig = mn2tn2wordtype2importancesig[mn]

    importances = pd.DataFrame(mn2tn2wordtype2importance[plot_mns[-1]])
    xlabels = [tn2name(tt) for tt in importances.keys()]
    ylabels = list(importances.index)
    values = importances.values

    fig, ax = plt.subplots()

    # Add white borders between columns
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    x_spacing = 0.1
    for xcoord in range(-1, len(xlabels)):
        ax.axvline(0.5 + xcoord, 0 - x_spacing, 1 + x_spacing, linewidth=4, zorder=10, color='k' if exp in {'pos', 'gender'} and xcoord == 6 else 'w')

    title = "MDL Increase from POS Masking over Random Masking"
    values = np.round(values, 1)
    colorvalues = values / np.abs(values).max(0)
    im = ax.imshow(colorvalues, cmap='RdBu')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            text_str = f'${round(values[i, j])}$'
            if not tn2wordtype2importancesig[xlabels[j].lower().replace('$_', '.round').strip('$')][ylabels[i]]:
                text_str = text_str[:-1] + '^*$'
            text = ax.text(j, i, text_str, ha="center", va="center", color="k" if abs(colorvalues[i, j]) < 0.9 else 'w', fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    save_and_show_plot(plt, f'{title.replace(" ", "_")}.heatmap.exp-{exp}.mn-{mn2name[mn]}', savefig)
