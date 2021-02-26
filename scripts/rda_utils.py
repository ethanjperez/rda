from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
import pathlib
import re
from scipy.stats import multivariate_normal
import subprocess
import torch
import warnings


# Configure GPU batch size for each model architecture (for *.from_scratch models, use the GPU batch size from the same architecture, e.g., roberta-base for roberta-base.from_scratch)
mn2max_tbs = {
    'roberta-base': 16,
    'roberta-large': 8,
    'distilroberta-base': 16,
    'allenai/longformer-base-4096': 4,
    'allenai/longformer-large-4096': 1,
    'albert-base-v2': 64,
    'albert-large-v2': 32,
    'albert-xlarge-v2': 16,
    'albert-xxlarge-v2': 16,
    'albert-base-v1': 64,
    'albert-large-v1': 32,
    'albert-xlarge-v1': 16,
    'albert-xxlarge-v1': 16,
    'xlnet-base-cased': 64,
    'xlnet-large-cased': 16,
    'facebook/bart-base': 32,
    'facebook/bart-large': 16,
    'distilgpt2': 64,
    'gpt2': 64,
    'gpt2-medium': 16,
    'gpt2-large': 8,
    'gpt2-xl': 2,
}


cm = plt.cm.plasma
temps = 10 ** np.linspace(-1, 2, 1000)
variances = 10 ** np.linspace(-2.5, 1.5, 10000)
temperature_filename = 'test_results.with_temperature.json'
PLOTS_DIR = f'{os.environ["BASE_DIR"]}/plots'


def load_losses(task_types, analysis_type, do_tune_temperature, cache_mn2tn2losses, seeds, model_names):
    """Load all codelengths from trained models (also tuning temperature)"""
    all_mn2tn2losses = {}
    for task_type in task_types:
        print(task_type)
        group = f'{task_type}.{analysis_type}' if len(analysis_type) > 0 else task_type
        tns = group2datasethps[group]['tn']
        task_type = group.split('.', 1)[0]

        max_num_samples = tn2max_num_samples(tns[0])
        for tn in tns:
            assert max_num_samples == tn2max_num_samples(tn), f'Expected tn2max_num_samples(tn) == {max_num_samples} but got {tn2max_num_samples(tn)}'

        mns = model_names if model_names is not None else deepcopy(group2datasethps[group]['mn'])
        ntss = tn2block_start_idxs(tns[0])
        pbns = list(range(len(ntss)))[:-1]

        regression = task_type.lower().startswith('sts-b')

        mn2tn2losses_file = f'{os.environ["BASE_DIR"]}/checkpoint/rda/group-{group}.mn2tn2losses.json'
        mn2tn2losses = None
        if cache_mn2tn2losses:
            try:
                with open(mn2tn2losses_file) as f:
                    mn2tn2losses = json.load(f)
                print('\tLoaded cached losses file!')
            except FileNotFoundError:  # Here, we'll load from scratch (not from cached file)
                pass
        if mn2tn2losses is None:
            assert (not regression) or do_tune_temperature, f'regression ({regression}) expects do_tune_temperature = True but got ({do_tune_temperature})'
            val_loss_select_best_key = 'test_loss' if 'clevr' in group else ('val_loss_no_temperature' if do_tune_temperature else 'val_loss')
            mn2tn2losses = {mn: {tn: [[] for _ in pbns] for tn in tns} for mn in mns}
            diagnostic2count = defaultdict(int)
            for mn in mns[::-1]:
                for tn in tqdm(tns, desc=mn):
                    cached_test_results_file = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{tn}.mn-{mn}.pbn2seed2hpstr2stats.json'
                    if not do_tune_temperature:
                        cached_test_results_file = cached_test_results_file.replace('.pbn2seed2hpstr2stats.json', '.no_temperature.pbn2seed2hpstr2stats.json')
                    if os.path.exists(cached_test_results_file):
                        with open(cached_test_results_file) as f:
                            pbn2seed2hpstr2stats = json.load(f)
                        pbn2seed2hpstr2stats = {int(pbn): {int(seed): hpstr2stats for seed, hpstr2stats in seed2hpstr2stats.items()} for pbn, seed2hpstr2stats in pbn2seed2hpstr2stats.items()}
                    else:
                        pbn2seed2hpstr2stats = {pbn: {seed: {} for seed in seeds} for pbn in pbns}
                        num_missing_stats = 0
                        for pbn in pbns:
                            for seed in seeds:
                                hpstrs = mn2hpstrs(mn)
                                if ('clevr' not in group) and (seed != seeds[0]) and ('bs-' in hpstrs[0]) and ('lr-' in hpstrs[0]) and ('nte-' in hpstrs[0]):
                                    hpstrs = [f'bs-{32 if tn.startswith("hotpot") else 0}.lr-0.nte-{hpstrs[0].split("nte-")[-1]}']

                                for hpstr in hpstrs:
                                    stats, error = load_stats(tn, mn, hpstr, seed, pbn, do_tune_temperature)
                                    if error is not None:  # Check for error loading stats
                                        diagnostic2count[error] += 1
                                        num_missing_stats += 1
                                    else:
                                        pbn2seed2hpstr2stats[pbn][seed][hpstr] = stats

                        if num_missing_stats == 0:
                            os.makedirs(cached_test_results_file.rsplit('/', 1)[0], exist_ok=True)
                            with open(cached_test_results_file, 'w') as f:
                                json.dump(pbn2seed2hpstr2stats, f)
                        else:
                            print(f'Missing {num_missing_stats} stats files. Not caching results for: tn-{tn}.mn-{mn}')

                    for pbn in pbns:
                        for seed in seeds:
                            hpstrs = mn2hpstrs(mn)
                            if ('clevr' not in group) and (seed != seeds[0]) and ('bs-' in hpstrs[0]) and ('lr-' in hpstrs[0]) and ('nte-' in hpstrs[0]):
                                hpstrs = [f'bs-{32 if tn.startswith("hotpot") else 0}.lr-0.nte-{hpstrs[0].split("nte-")[-1]}']

                            best_stats = {}
                            for hpstr in hpstrs:
                                stats = pbn2seed2hpstr2stats[pbn][seed][hpstr]
                                if stats[val_loss_select_best_key] == float('nan'):
                                    diagnostic2count['bad_run'] += 1  # if bad val loss
                                elif stats[val_loss_select_best_key] < best_stats.get(val_loss_select_best_key, float('inf')):
                                    best_stats = stats  # if best val loss

                            if best_stats.get(val_loss_select_best_key, float('inf')) == float('inf'):
                                print(f'No files found for tn-{tn}.mn-{mn}.pbn-{pbn}.train.log (seed-{seed})')
                                continue
                            assert best_stats['test_loss'] == best_stats['avg_test_loss'], f"Expected best_stats['test_loss'] ({best_stats['test_loss']}) == best_stats['avg_test_loss'] ({best_stats['avg_test_loss']})"
                            mn2tn2losses[mn][tn][pbn].append(best_stats['avg_test_loss'])

            if cache_mn2tn2losses:
                with open(mn2tn2losses_file, 'w') as f:
                    json.dump(mn2tn2losses, f)
                print('Cached losses!\n')

        for mn, tn2losses in mn2tn2losses.items():
            if mn not in all_mn2tn2losses:
                all_mn2tn2losses[mn] = {}
            for tn, losses in tn2losses.items():
                if tn in all_mn2tn2losses[mn]:
                    assert losses == all_mn2tn2losses[mn][tn], 'Expected pre-loaded and newly loaded losses to be equal!'
                all_mn2tn2losses[mn][tn] = losses
    return all_mn2tn2losses


def save_and_show_plot(plt_obj, filebase, savefig=True):
    if savefig:
        plt_obj.savefig(f'{PLOTS_DIR}/{filebase}.png', dpi=900, bbox_inches='tight')
        plt_obj.savefig(f'{PLOTS_DIR}/{filebase}.pdf', bbox_inches='tight')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt_obj.show()
    plt.close()


def get_mdls(tn2lossmatrix, mdl_tns, block_size, baseline_loss=None):
    mdls = []
    num_seeds = tn2lossmatrix[mdl_tns[0]].shape[1]
    for seed_idx in range(num_seeds):
        lossmeans = [tn2lossmatrix[tn][:, seed_idx] for tn in mdl_tns]
        if baseline_loss is not None:
            lossmeans.append(baseline_loss * np.ones_like(tn2lossmatrix[mdl_tns[0]][:, 0]))
        bits_per_block = np.array(lossmeans) * block_size
        mdls.append(bits_per_block.sum(axis=1))
    return np.vstack(mdls).transpose()


def get_ablation2mdl(tns, mn, mn2tn2lossmatrix, tn_pattern='.shuffle'):
    tn2mdlstats = {}

    ntss = tn2block_start_idxs(tns[0])
    block_starts = np.array([0] + ntss[:-1].tolist())
    block_ends = ntss
    block_size = block_ends - block_starts
    task_type = tns[0].split(".", 1)[0] if not tns[0].startswith('anli.round') else '.'.join(tns[0].split(".", 2)[:2])

    tn_matches = [tn for tn in tns if tn_pattern in tn]
    assert len(tn_matches) == 1, f'Expected len(tn_matches) ({len(tn_matches)}) == 1'
    ablation_name = tn_matches[0]
    compared_tns = [tns[0], ablation_name]
    mdls = get_mdls(mn2tn2lossmatrix[mn], compared_tns, block_size, baseline_loss=tt2entropy[task_type] / np.log(2))
    assert mdls.shape[1] == 5, f'Usually expecting 5 random seeds (remove assertion to override)'
    for i in range(mdls.shape[0]):
        if i == (mdls.shape[0] - 1):
            table_name = '$\mathcal{H}(y)$'
        elif i == 0:
            table_name = 'Original'
        else:
            table_name = tn2name(compared_tns[i])
            if table_name == 'Shuffle':
                table_name = 'Shuffled'
            if table_name == 'Length':
                table_name = 'Length Only'
        mean = mdls[i].mean()
        stderr = mdls[i].std() / (mdls.shape[1] ** 0.5)
        tn2mdlstats[table_name] = (mean, stderr)
    tn2mdlstats = {tn: (mean / tn2mdlstats['$\mathcal{H}(y)$'][0], stderr / tn2mdlstats['$\mathcal{H}(y)$'][0])
                   for tn, (mean, stderr) in tn2mdlstats.items()}
    return tn2mdlstats


def plot_mdls(tns, mn, group, mn2tn2lossmatrix, savefig=True, figsize=None, broken_axis=False, override_cm=None, xlabels_rotation=30, extra_bits=0, first_n_broken=float('inf'), plot_kilobits=True):
    ntss = tn2block_start_idxs(tns[0])
    block_starts = np.array([0] + ntss[:-1].tolist())
    block_ends = ntss
    block_size = block_ends - block_starts
    plot_cm = override_cm if override_cm is not None else cm

    if broken_axis:
        fig, (ax_top, ax) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [15, 1]})
        hspace = 0.05
        fig.subplots_adjust(hspace=hspace)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    bottom_max_mdl = 10
    top_min_mdl = float('inf')
    top_max_mdl = float('-inf')
    tn2mdls = {}
    labels = []
    for tn_no, tn in enumerate(tns):
        losses = mn2tn2lossmatrix[mn][tn]
        bits_per_block = losses * block_size.reshape(-1, 1)
        mdls = bits_per_block.sum(axis=0) + extra_bits
        tn2mdls[tn] = mdls
        if plot_kilobits:
            mdls = mdls / 1000.
        labels.append(tn2name(tn))
        mdl_stderr = (mdls.std() / (len(mdls) ** 0.5))
        if broken_axis:
            ax_top.bar(len(labels) - 1, mdls.mean(), color=plot_cm(tn_no / len(tns)), yerr=mdl_stderr)
        ax.bar(len(labels) - 1, mdls.mean(), color=plot_cm(tn_no / len(tns)), yerr=mdl_stderr)
        if tn_no < first_n_broken:
            top_min_mdl = min(top_min_mdl, mdls.mean() - mdl_stderr)
            top_max_mdl = max(top_max_mdl, mdls.mean() + mdl_stderr)
        else:
            bottom_max_mdl = max(bottom_max_mdl, mdls.mean() + mdl_stderr) * 1.03
    if broken_axis:
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.get_yaxis().set_visible(False)
        top = top_max_mdl * 1.03
        ax_top.set_ylim(bottom=min(top - bottom_max_mdl, top_min_mdl * (.5 if group.lower().startswith('clevr') else .95)), top=top)
        ax.set_ylim(top=bottom_max_mdl)

        ax_top.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
        ax.xaxis.tick_bottom()

        d = .025  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d-hspace), (-d, +d-hspace), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d-hspace), (-d, +d-hspace), **kwargs)  # top-right diagonal

    ax.set_xticks(np.array(range(len(labels))))
    ax.set_xticklabels(labels, rotation=xlabels_rotation, ha='right')
    bits_name = f'{"kilo" if plot_kilobits else ""}bits'
    if broken_axis:
        ax_top.set_ylabel(f'MDL ({bits_name})', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax_top.tick_params(axis='both', which='major', labelsize=17)
    else:
        ax.set_ylabel(f'MDL ({bits_name})', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=18)
    save_and_show_plot(plt, f'group-{group}.mn-{mn2name[mn]}', savefig)
    return tn2mdls


def get_ablation2mdlincrease(tns, mn, mn2tn2lossmatrix, tn_patterns=('.shuffle', '.length', '.mask_all')):
    ablation2mdlincrease = {}
    ablation2mdlincreasestderr = {}

    ntss = tn2block_start_idxs(tns[0])
    block_starts = np.array([0] + ntss[:-1].tolist())
    block_ends = ntss
    block_size = block_ends - block_starts
    for tn in tns:
        matching_tags = [tag for tag in tn_patterns if tag in tn]
        if len(matching_tags) == 0:
            continue
        ablation_name = matching_tags[0]
        mdls = get_mdls(mn2tn2lossmatrix[mn], [tns[0], tn], block_size)
        ablation2mdlincrease[ablation_name] = (mdls[1] - mdls[0]).mean()
        ablation2mdlincreasestderr[ablation_name] = (mdls[1] - mdls[0]).std() / (mdls.shape[1] ** 0.5)
        assert mdls.shape[1] == 5, f'Usually expecting 5 random seeds (remove assertion to override)'
    return ablation2mdlincrease, ablation2mdlincreasestderr


def plot_loss_data_curve_prequential(tns, mn, group, mn2tn2lossmatrix, savefig=True, legend_params={'bbox_to_anchor': (1, 1)}, log_log_scale=False, use_mn_in_ylabel=False, short_ylabel=False, override_cm=None, crop_first_block=False):
    task_type = group.split(".", 1)[0] if not group.startswith('anli.round') else '.'.join(group.split(".", 2)[:2])
    ntss = tn2block_start_idxs(tns[0])
    block_starts = np.array([0] + ntss[:-1].tolist())
    block_ends = ntss
    plot_cm = override_cm if override_cm is not None else cm

    fig, ax = plt.subplots()
    if crop_first_block:
        ax.set_ylim(top=max([loss for tn in tns for loss in mn2tn2lossmatrix[mn][tn][1]]) * 1.05)
    for tn_no, tn in enumerate(tns):
        losses = mn2tn2lossmatrix[mn][tn]
        xs = []
        ys = []
        ys_upper = []
        ys_lower = []
        for bs, be, l in zip(1 + block_starts, block_ends, losses):
            xs += [bs, be]
            ymid = l.mean()
            yerr = (l.std() / (len(l) ** 0.5))
            yupper = ymid + yerr
            ylower = ymid - yerr
            ys += [ymid] * 2
            ys_upper += [yupper] * 2
            ys_lower += [ylower] * 2
        plt.plot(xs, ys, color=plot_cm(tn_no / len(tns)), label=tn2name(tn), linestyle='solid')
        plt.fill_between(xs, ys_lower, ys_upper, color=plot_cm(tn_no / len(tns)), alpha=0.3, linewidth=0)
    xmin = 1
    if log_log_scale:
        plt.xscale('log')
        plt.yscale('log')
        xmin = ntss[0] / (ntss[1] / ntss[0])
    else:
        plt.ylim(bottom=0)
    if not crop_first_block:
        plt.hlines(tt2entropy[task_type] / np.log(2), xmin=xmin, xmax=ntss[-1], linestyles='dashed', label='$H(y)$', color='black')
    plt.xlim([xmin, ntss[-1]])
    plt.legend(**legend_params)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel('Number of training samples', fontsize=20)

    # plot ylabel
    tn_name = tn2name(group.split(".", 1)[0]).replace(' ', '~')
    ylabel_mn = r' $\mathbf{(' + mn2name[mn].strip("$") + r')}$' if use_mn_in_ylabel else ''
    sep = ": " if short_ylabel else "\n"
    ylabel = f'$\mathbf{{{tn_name.replace("-", "")}}}${ylabel_mn}{sep}Codelength per sample'
    if not short_ylabel:
        ylabel += ' (bits)'
    plt.ylabel(ylabel, fontsize=15)

    plt.tight_layout()
    save_and_show_plot(plt, f'loss_data_curve.group-{group}.mn-{mn2name[mn]}', savefig)


def plot_wordtype_importances(tns, mn, mn2tn2lossmatrix, group, savefig=True, exp=None):
    ntss = tn2block_start_idxs(tns[0])
    block_starts = np.array([0] + ntss[:-1].tolist())
    block_ends = ntss
    block_size = (block_ends - block_starts)

    tn_pairs = []
    for tn in tns:
        if exp == 'gender':
            if ('_male_words' in tn) and ('_fraction' not in tn):
                tn_pairs.append([tn, tn.replace('_male_words', '_female_words')])
        else:
            if '_fraction' in tn:
                tn_pairs.append([tn.replace('_fraction', ''), tn])

    fig, ax = plt.subplots()
    wordtype2importance = {}
    wordtype2importancestderr = {}
    labels = []
    for tn_pair_no, (tn_ablation, tn_ablation_frac) in enumerate(tn_pairs):
        mdls = get_mdls(mn2tn2lossmatrix[mn], [tn_ablation, tn_ablation_frac], block_size)
        mdl_diff = (mdls[0] - mdls[1]).mean()
        mdl_diff_stderr = (mdls[0] - mdls[1]).std() / (mdls.shape[1] ** 0.5)
        assert mdls.shape[1] == 5, f'Usually expecting 5 random seeds (remove assertion to override)'
        plt.bar(tn_pair_no, mdl_diff, color=cm(tn_pair_no / len(tn_pairs)), yerr=mdl_diff_stderr)
        label = tn2name(tn_ablation).replace('Mask ', '').replace('POS ', '')
        labels.append(label)
        wordtype2importance[label] = mdl_diff
        wordtype2importancestderr[label] = mdl_diff_stderr
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.title(f'Importance of Different POS Words for {tn2name(tns[0])}')
    plt.ylabel('$MDL_{ablation}$ - $MDL_{baseline}$', fontsize=14)
    save_and_show_plot(plt, f'increase_by_pos.group-{group}.mn-{mn2name[mn]}', savefig)
    return wordtype2importance, wordtype2importancestderr


def get_mn2tn2lossmatrix(mns, tns, task_type, mn2tn2losses):
    ntss = tn2block_start_idxs(tns[0])
    max_loss = tt2max_loss[task_type] / np.log(2)
    mn2tn2lossmatrix = {}
    for mn in mns:
        mn2tn2lossmatrix[mn] = {}
        tn2losses = mn2tn2losses[mn]
        for tn_no, tn in enumerate(tns):
            losses = np.array(tn2losses[tn])[:len(ntss)-1] / np.log(2)
            losses = np.concatenate([max_loss * np.ones((1, losses.shape[1])), losses], axis=0)
            mn2tn2lossmatrix[mn][tn] = losses
    return mn2tn2lossmatrix


def plot_loss_reduction_by_loss_with_error(tns, mn, group, tn2lossmatrix, savefig=True):
    baseline_loss = tn2lossmatrix[tns[0]][1:]
    baseline_loss_mean = baseline_loss.mean(1)
    fig, ax = plt.subplots()
    plt.xlim([min(baseline_loss_mean), max(baseline_loss_mean)])
    for tn_no, tn in enumerate(tns):
        tn_loss = tn2lossmatrix[tn][1:]
        loss_reduction = 100. * (1 - (tn_loss / baseline_loss))
        mean_loss_reduction = loss_reduction.mean(1)
        stderr_loss_reduction = loss_reduction.std(1) / np.sqrt(loss_reduction.shape[1])
        plt.plot(baseline_loss_mean, mean_loss_reduction, label=tn2name(tn), color=cm(tn_no / len(tns)))
        plt.fill_between(baseline_loss_mean, mean_loss_reduction - stderr_loss_reduction, mean_loss_reduction + stderr_loss_reduction, color=cm(tn_no / len(tns)), alpha=0.3, linewidth=0)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title(f'{tn2name(group)} ({mn2name[mn]})', fontsize=16)
    plt.ylabel('% Codelength Reduction\nOver No Decomposition', fontsize=16)
    plt.xlabel('No Decomposition Codelength (bits)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(fontsize=11.5)
    plt.tight_layout()
    save_and_show_plot(plt, f'loss_reduction_by_loss_with_error.group-{group}.mn-{mn2name[mn]}', savefig)


def read_results(save_dir, filename=None, read_json=False):
    if read_json:
        with open(os.path.join(save_dir, filename if ((filename is not None) and (filename.endswith('.json'))) else 'results.json')) as f:
            stats = json.load(f)
        if 'test_loss' in stats:
            stats['avg_test_loss'] = stats['test_loss']
        elif 'avg_test_loss' in stats:
            stats['avg_test_loss'] = stats['avg_test_loss']
        else:
            test_loss_keys = [k for k in stats if k.startswith('loss_')]
            assert len(test_loss_keys) == 1, f'len(test_loss_keys) == {len(test_loss_keys)} != 1'
            stats['avg_test_loss'] = stats[test_loss_keys[0]]
    else:
        stats = {}
        with open(os.path.join(save_dir, f'test_results.txt' if filename is None else filename)) as f:
            for line in f:
                k, v = line.strip().split(' = ')
                stats[k] = float(v.split('tensor(', 1)[-1].rstrip(')').split(',', 1)[0]) if v.startswith('tensor(') else float(v)
    return stats


def compute_gaussian_nll(labels, preds, variance):
    return float(-np.log(multivariate_normal.pdf(labels - preds, mean=0, cov=variance)).mean())


def tune_std(df, verbose=False):
    stats = {}
    split = 'dev'
    preds = df[split][df[split].keys()[0]].values
    labels = df[split]['label'].values
    variance2loss = {variance: compute_gaussian_nll(labels, preds, variance) for variance in variances}
    stats['temperature'] = sorted(variance2loss, key=variance2loss.get)[0]
    stats['val_loss'] = variance2loss[stats['temperature']]
    if verbose:
        print(f'Best Temperature: {stats["temperature"]} / Val Loss: {stats["val_loss"]} (Val Loss = {compute_gaussian_nll(labels, preds, 1)} @ T=1)')
    split = 'test'
    preds = df[split][df[split].keys()[0]].values
    labels = df[split]['label'].values
    stats['test_loss'] = compute_gaussian_nll(labels, preds, stats['temperature'])
    stats['avg_test_loss'] = stats['test_loss']
    return stats


def tune_temperature(df, verbose=False):
    loss_func = torch.nn.NLLLoss()
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    stats = {}
    split = 'dev'
    logprobs = torch.from_numpy(np.log(df[split][df[split].keys()[:-1]].values))
    labels = torch.from_numpy(df[split]['label'].values)
    temp2loss = {temp: loss_func(logsoftmax(logprobs / temp), labels).item() for temp in temps}
    stats['temperature'] = sorted(temp2loss, key=temp2loss.get)[0]
    stats['val_loss'] = temp2loss[stats['temperature']]
    stats['val_acc'] = (logprobs.argmax(dim=1) == labels).float().mean().item()
    if verbose:
        print(f'Best Temperature: {stats["temperature"]} / Val Loss: {stats["val_loss"]}')

    split = 'test'
    logprobs = torch.from_numpy(np.log(df[split][df[split].keys()[:-1]].values))
    labels = torch.from_numpy(df[split]['label'].values)
    stats['test_loss'] = loss_func(logsoftmax(logprobs / stats['temperature']), labels).item()
    stats['test_acc'] = (logprobs.argmax(dim=1) == labels).float().mean().item()
    stats['avg_test_loss'] = stats['test_loss']
    return stats


def adjust_preds_with_temperature(df, temperature):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    logprobs = torch.from_numpy(np.log(df[df.keys()[:-1]].values))
    logsoftmax_preds = logsoftmax(logprobs / temperature)
    df_temp = pd.DataFrame(np.exp(logsoftmax_preds.numpy()), columns=df.keys()[:-1])
    df_temp['label'] = df['label']
    return df_temp


def save_tuned_stats(save_dir):
    regression = 'tn-sts-b' in save_dir.lower()
    stats = {f'{k}_no_temperature': v for k, v in read_results(save_dir).items()}
    val_preds = pd.read_csv(f'{save_dir}/preds.val.epoch-{int(stats["best_val_epoch_no_temperature"])}.csv')
    test_preds = pd.read_csv(f'{save_dir}/preds.test.epoch-0.csv')
    stats.update((tune_std if regression else tune_temperature)({'dev': val_preds, 'test': test_preds}))
    with open(f'{save_dir}/{temperature_filename}', 'w') as f:
        json.dump(stats, f)
    return stats


def save_temperature_results(save_dir, return_stats=False):
    if 'mn-fasttext' in save_dir:
        if not return_stats:
            return
        stats = read_results(save_dir)
        stats['best_epoch'] = None
        return stats

    if os.path.exists(f'{save_dir}/{temperature_filename}'):
        try:
            with open(f'{save_dir}/{temperature_filename}') as f:
                stats = json.load(f)
        except json.JSONDecodeError:
            print('JSONDecodeError!')
            save_tuned_stats(save_dir)
    else:
        stats = save_tuned_stats(save_dir)

    if return_stats:
        return stats


def classify_log_error_and_time(log_file):
    if not os.path.exists(log_file):
        return 'missing_log_file', None

    error_time = datetime.fromtimestamp(round(pathlib.Path(log_file).stat().st_mtime))
    lines = []
    from sh import tail
    for line in tail("-10", log_file, _iter=True):
        if "CUDA error: out of memory" in line:
            return 'cuda_oom', error_time
        elif "RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable" in line:
            return 'cuda_unavailable', error_time
        elif "KeyError" in line:
            return 'key_error', error_time
        elif 'NODE FAILURE' in line:
            return 'node_failure', error_time
        elif 'NotImplementedError: Model Name' in line:
            return 'model_name', error_time
        elif 'AssertionError: No validation results read from previous runs!' in line:
            return 'no_validation_runs_read', error_time
        elif 'core dumped' in line:
            return 'core_dumped', error_time
        lines.append(line.strip())

    if ('Epoch' in lines[-1]) or ('Testing' in lines[-1]) or ('Progress:' in lines[-1]) or ('it/s' in lines[-1]):
        cmd = """squeue -o "%.1000j" -u ejp416"""
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, err = process.communicate()
        return 'running' if log_file in str(output) else 'cpu_oom?', error_time

    print(f'\tFileNotFoundError when loading results for {log_file}')
    print('\n'.join(lines))
    return 'other', error_time


def get_diagnostic(save_dir, log_file):
    if not os.path.exists(save_dir):
        assert 'fasttext' not in save_dir, f'Should not be missing fasttext model directories like {save_dir}'
        return 'missing_dir'
    diagnostic, error_time = classify_log_error_and_time(log_file)
    print(diagnostic, error_time, log_file)
    return diagnostic


def load_stats(tn, mn, hpstr, seed, pbn, do_tune_temperature):
    save_dir = f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{tn}.mn-{mn}.{hpstr}.seed-{seed}.pbn-{pbn}'
    log_file = f'{save_dir}/train.log' if (tn.startswith('hotpot') or tn.startswith('clevr')) else f'{os.environ["BASE_DIR"]}/checkpoint/rda/tn-{tn}.mn-{mn}.pbn-{pbn}.train.log'
    if mn == 'FiLM':
        try:
            stats = read_results(save_dir)
        except FileNotFoundError:
            print(f'Error loading CLEVR results. See Cassio file: {log_file}')
            return None, 'clevr'
    else:
        try:
            read_json = tn.startswith('hotpot')
            stats = save_temperature_results(save_dir, return_stats=True) if do_tune_temperature else read_results(save_dir, read_json=read_json)
            stats['test_loss'] = stats['avg_test_loss']
        except FileNotFoundError:
            return None, get_diagnostic(save_dir, log_file)
    stats['hpstr'] = hpstr
    stats['save_dir'] = save_dir
    return stats, None


def mn2hps(mn):
    if 'longformer' in mn:
        return {
            'bs': [32],
            'lr': ['3e-5', '5e-5', '1e-4'],
            'nte': [6],
        }
    elif 'roberta' in mn:
        if 'from_scratch' in mn:
            return {
                'bs': [64, 128] if 'large' in mn else [16, 32],
                'lr': ['1e-5', '2e-5', '3e-5'],
                'nte': [10],
            }
        else:
            return {
                'bs': [16, 32],
                'lr': ['1e-5', '2e-5', '3e-5'],
                'nte': [10],
            }
    elif 'fasttext' in mn:
        return {'ad': [7200]}  # Old: 21600
    elif 'facebook/bart' in mn:
        return {
            'bs': [32, 128],
            'lr': ['5e-6', '1e-5', '2e-5'],
            'nte': [10],
        }
    elif ('albert' in mn) or ('xlnet' in mn):
        return {
            'bs': [32, 128],
            'lr': ['2e-5', '3e-5', '5e-5'],
            'nte': [3],
        }
    elif 'gpt' in mn:
        return {
            'bs': [32],
            'lr': ['6.25e-5', '3.125e-5', '1.25e-4'],
            'nte': [3],
        }
    elif 'FiLM' in mn:
        return {
            'bs': [64],
            'lr': ['3e-4'],
            'nte': [20]
        }
    else:
        raise NotImplementedError(f'Model Name = {mn}')


def mn2hpstrs(mn):
    hp2value = mn2hps(mn)
    hpstrs = ['']
    for hp in sorted(hp2value.keys()):
        hpstrs = [f'{hpstr}{"." if len(hpstr) > 0 else ""}{hp}-{hp_value}' for hpstr in hpstrs for hp_value in hp2value[hp]]
    return hpstrs


def tn2max_num_samples(tn):
    if tn.startswith('hotpot'):
        return 90446
    elif tn.startswith('eanli') and tn.endswith('round3'):
        return 8447
    elif tn.startswith('esnli') or tn.startswith('nli.snli'):
        return 549367
    elif tn.startswith('mnli') or tn.startswith('nli.mnli'):
        return 392702
    elif tn.startswith('qnli') or tn.startswith('nli.qnli'):
        return 104743
    elif tn.startswith('wnli') or tn.startswith('nli.wnli'):
        return 635
    elif tn.startswith('anli.round1'):
        return 16946
    elif tn.startswith('anli.round2'):
        return 45460
    elif tn.startswith('anli.round3'):
        return 100459
    elif tn.startswith('questions.comparison') or tn.startswith('clevr.comparison'):
        return 83105
    elif tn.startswith('questions.compare_integer') or tn.startswith('clevr.compare_integer'):
        return 15942
    elif tn.startswith('questions.same_relate') or tn.startswith('clevr.same_relate'):
        return 82604
    elif 'cola' in tn:
        return 8551
    elif 'mrpc' in tn:
        return 3668
    elif 'qqp' in tn:
        return 363849  # NB: Only includes read-in examples (full dataset has 363870 examples)
    elif 'rte' in tn:
        return 2490
    elif 'sst-2' in tn:
        return 67349
    elif 'sts-b' in tn:
        return 5749
    else:
        raise NotImplementedError(f'Task Name = {tn}')


def tn2block_start_idxs(tn, prequential_num_blocks=8, prequential_min_block_size=64, prequential_max_block_size=10000):
    max_num_samples = tn2max_num_samples(tn)
    prequential_max_num_train_samples = min(prequential_max_block_size, max_num_samples) if not (tn.startswith('hotpot') or tn.startswith('clevr')) else max_num_samples
    return np.round(np.logspace(np.log10(prequential_min_block_size), np.log10(prequential_max_num_train_samples), prequential_num_blocks + 1)).astype(int)


mn2name = {
    'ensemble': 'Ensemble Model',
    'min-loss': 'Ensemble Model',
    'average-prob': 'Ensemble Model',
    'average-logprob': 'Ensemble Logit Model',
    'average-prob-with-temperature': 'Ensemble Model',
    'average-logprob-with-temperature': 'Ensemble Logit Model',
    'roberta-base': '$RoBERTa_{BASE}$',
    'roberta-large': '$RoBERTa_{LARGE}$',
    'roberta-base.from_scratch': '$Transformer_{BASE}$',
    'roberta-large.from_scratch': '$Transformer_{LARGE}$',
    'distilroberta-base': '$DistilRoBERTa$',
    'allenai/longformer-base-4096': '$Longformer_{BASE}$',
    'allenai/longformer-base-4096.from_scratch': '$Transformer_{BASE}$',
    'allenai/longformer-large-4096': '$Longformer_{LARGE}$',
    'allenai/longformer-large-4096.from_scratch': '$Transformer_{LARGE}$',
    'fasttext': 'fastText',
    'fasttext_no_pretrain': '$fastText (from scratch)$',
    'fasttext_sent1_upper': '$fastText (1st Sent. Upper Case)$',
    'fasttext_sent2_upper': '$fastText (2nd Sent. Upper Case)$',
    'albert-base-v2': '$ALBERT_{BASE}$',
    'albert-large-v2': '$ALBERT_{LARGE}$',
    'albert-xlarge-v2': '$ALBERT_{XLARGE}$',
    'albert-xxlarge-v2': '$ALBERT_{XXLARGE}$',
    'albert-base-v1': '$ALBERT_{BASE} (v1)$',
    'albert-large-v1': '$ALBERT_{LARGE} (v1)$',
    'albert-xlarge-v1': '$ALBERT_{XLARGE} (v1)$',
    'albert-xxlarge-v1': '$ALBERT_{XXLARGE} (v1)$',
    'xlnet-base-cased': '$XLNet_{BASE}$',
    'xlnet-large-cased': '$XLNet_{LARGE}$',
    'facebook/bart-base': '$BART_{BASE}$',
    'facebook/bart-large': '$BART_{LARGE}$',
    'gpt2': '$GPT2$',
    'gpt2-medium': '$GPT2 Medium$',
    'gpt2-large': '$GPT2 Large$',
    'gpt2-xl': '$GPT2 XL$',
    'distilgpt2': '$DistilGPT2$',
    'FiLM': '$FiLM$'
}


tn2namedict = {
    # (e)ANLI
    'anli.round1': 'ANLI$_1$',
    'anli.round2': 'ANLI$_2$',
    'anli.round3': 'ANLI$_3$',
    "eanli.round3": 'eANLI',
    "eanli-reason.round3": 'eANLI (+Reason)',
    "eanli-wrong-prediction.round3": 'eANLI (+Wrong Pred)',
    "eanli-reason-wrong-prediction.round3": 'eANLI (+Reason+Wrong Pred)',
    "eanli-random-wrong-prediction.round3": 'eANLI (+Random Wrong Pred)',

    # (e)SNLI
    'snli': 'SNLI',
    'esnli': 'e-SNLI',
    'esnli.input-raw': 'Input (I)',
    'esnli.input-raw,explanation': 'I+E',
    'esnli.input-marked': 'I+R',
    'esnli.input-marked,explanation': 'I+R+E',
    'esnli.input-explanation': 'Expl. (E)',
    'esnli.input-markedonly': 'Rationale (R)',
    'esnli.input-markedmasked': 'Non-R Words',
    'esnli.input-markedunmasked': 'R Words',

    # GLUE
    'cola': 'CoLA',
    'sst-2': 'SST-2',
    'mrpc': 'MRPC',
    'sts-b': 'STS-B',
    'qqp': 'QQP',
    'mnli': 'MNLI',
    'qnli': 'QNLI',
    'wnli': 'WNLI',
    'rte': 'RTE',

    # HotpotQA
    'hotpot': 'HotpotQA',
    'hotpot.context-long.num_subas-0.shuffled-1': 'No SubQ',
    'hotpot.context-long.num_subas-1.shuffled-1': '+1 Oracle',
    'hotpot.context-long.num_subas-2.shuffled-1': 'Oracle',
    'hotpot.context-long.subqs-20639223.num_subas-1.shuffled-1': '+1 ONUS',
    'hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1': 'ONUS',
    'hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1.use_subqs': 'ONUS SQs/SAs',
    'hotpot.context-long.subqs-7.num_subas-2.shuffled-1': 'DLM',
    'hotpot.context-long.subqs-20604919.num_subas-2.shuffled-1': 'Seq2Seq',
    'hotpot.context-long.subqs-21979200.num_subas-2.shuffled-1': 'PseudoD',
    'hotpot.context-long.subqs-22008128.num_subas-2.shuffled-1': 'ONUS+Random',
    'hotpot.context-long.subqs-22026601.num_subas-2.shuffled-1': 'Seq2Seq+Random',
    'hotpot.context-long.subqs-22165087.num_subas-2.shuffled-1': 'PseudoD+Random',

    # CLEVR
    'questions.comparison.num_sas-0': 'Q+Image',
    'questions.comparison.num_sas-1': '+1 Oracle SA',
    'questions.comparison.num_sas-2': '+2 Oracle SAs',
    'questions.compare_integer.num_sas-0': 'Q+Image',
    'questions.compare_integer.num_sas-1': '+1 Oracle SA',
    'questions.compare_integer.num_sas-2': '+2 Oracle SAs',
    'questions.same_relate.num_sas-0': 'Q+Image',
    'questions.same_relate.num_sas-1': '+1 Oracle SA',
    'clevr-comparison': 'Attribute Comparison Q\'s',
    'clevr.comparison.num_sas-0': 'Q+Image',
    'clevr.comparison.num_sas-1': '+1 Oracle SA',
    'clevr.comparison.num_sas-2': '+2 Oracle SAs',
    'clevr-compare_integer': 'Integer Comparison Q\'s',
    'clevr.compare_integer.num_sas-0': 'Q+Image',
    'clevr.compare_integer.num_sas-1': '+1 Oracle SA',
    'clevr.compare_integer.num_sas-2': '+2 Oracle SAs',
    'clevr-same_relate': 'Same Property As Q\'s',
    'clevr.same_relate.num_sas-0': 'Q+Image',
    'clevr.same_relate.num_sas-1': '+1 Oracle SA',
}


def tn2name(tn):
    if re.search('\.v[0-9]', tn) is not None:  # remove version number
        tn = tn.rsplit('.v', 1)[0]
    if tn in tn2namedict:  # use dictionary name if given
        return tn2namedict[tn]
    else:  # otherwise assume it's an ablation and format the ablation name
        ablation_str = tn.rsplit('.', 1)[-1]
        ablation_words = []
        for word in ablation_str.split('_'):
            if word.lower() in {'pos'}:
                ablation_words.append(word.upper())
            else:
                ablation_words.append(word.capitalize())
        return ' '.join(ablation_words)


# NB: Not adding prior counts when computing entropy
tt2entropy = {
    'anli.round1': 1.0815673677121591,
    'anli.round2': 1.0549672881894927,
    'anli.round3': 1.0851179373820787,
    "clevr-comparison": 0.6929541100171355,
    "clevr-compare_integer": 0.6929250567534011,
    "clevr-same_relate": 1.2550521380047979,
    'hotpot': 8.986601996521237,  # 13.05210210128934 with prior_count = 1
    "esnli": 1.098611200899857,
    "snli": 1.098611200899857,
    "nli": 1.0986122885838119,  # NB: using MNLI value here
    "mnli": 1.0986122885838119,
    "cola": 0.6071229049995208,
    "sst-2": 0.6864445795502199,
    "mrpc": 0.6309590119522384,
    "sts-b": 1.8417526595092333,
    "qqp": 0.6585946744634452,
    "qnli": 0.6931471750454553,
    "wnli": 0.6929971327549332,
    "rte": 0.6931420193439887,
}
tt2max_loss = {
    'anli.round1': -np.log(1/3),
    'anli.round2': -np.log(1/3),
    'anli.round3': -np.log(1/3),
    "clevr-comparison": -np.log(1/2),
    "clevr-compare_integer": -np.log(1/2),
    "clevr-same_relate": -np.log(1/27),
    'hotpot': -np.log(1/((3450-10+1)*(170-0+1))),
    "esnli": -np.log(1/3),
    "snli": -np.log(1/3),
    "nli": -np.log(1/3),
    "mnli": -np.log(1/3),
    "cola": -np.log(1/2),
    "sst-2": -np.log(1/2),
    "mrpc": -np.log(1/2),
    "sts-b": -np.log(1. / (5. - 0.)),
    "qqp": -np.log(1/2),
    "qnli": -np.log(1/2),
    "wnli": -np.log(1/2),
    "rte": -np.log(1/2),
}

all_models = ['roberta-base', 'roberta-large', 'roberta-base.from_scratch', 'roberta-large.from_scratch', 'distilroberta-base', 'facebook/bart-base', 'albert-base-v2', 'gpt2', 'distilgpt2', 'fasttext']
group2datasethps = {
    'clevr-comparison': {
        'mn': ['FiLM'],
        'tn': [
            'clevr.comparison.num_sas-0',
            'clevr.comparison.num_sas-1',
            'clevr.comparison.num_sas-2',
        ],
    },
    'clevr-compare_integer': {
        'mn': ['FiLM'],
        'tn': [
            'clevr.compare_integer.num_sas-0',
            'clevr.compare_integer.num_sas-1',
            'clevr.compare_integer.num_sas-2',
        ],
    },
    'clevr-same_relate': {
        'mn': ['FiLM'],
        'tn': [
            'clevr.same_relate.num_sas-0',
            'clevr.same_relate.num_sas-1',
        ],
    },
    'hotpot': {
        'mn': ['allenai/longformer-base-4096', 'allenai/longformer-base-4096.from_scratch'],
        'tn': [
            'hotpot.context-long.num_subas-0.shuffled-1',
            # 'hotpot.context-long.subqs-22008128.num_subas-2.shuffled-1',
            # 'hotpot.context-long.subqs-22026601.num_subas-2.shuffled-1',
            # 'hotpot.context-long.subqs-22165087.num_subas-2.shuffled-1',
            # 'hotpot.context-long.num_subas-1.shuffled-1',
            'hotpot.context-long.subqs-21979200.num_subas-2.shuffled-1',
            'hotpot.context-long.subqs-20604919.num_subas-2.shuffled-1',
            # 'hotpot.context-long.subqs-20639223.num_subas-1.shuffled-1',
            # 'hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1.use_subqs',
            'hotpot.context-long.subqs-20639223.num_subas-2.shuffled-1',
            'hotpot.context-long.subqs-7.num_subas-2.shuffled-1',
            'hotpot.context-long.num_subas-2.shuffled-1',
        ],
    },
    'esnli': {
        'mn': all_models,
        'tn': [
            'esnli.input-raw',
            'esnli.input-markedonly',
            'esnli.input-markedmasked',
            'esnli.input-markedunmasked',
            'esnli.input-marked',
            'esnli.input-explanation',
            'esnli.input-raw,explanation',
            # 'esnli.input-marked,explanation',
        ],
    },
}


def get_taskname(task, tn_ending):
    if tn_ending == '':
        return 'esnli.input-raw' if task == 'snli' else task
    elif tn_ending != '' and task in {'mnli', 'snli'}:
        return f'nli.{task}.{tn_ending}.v2'
    else:
        return f'{task}.{tn_ending}'


glue_nli_tasks = ('cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'rte', 'qnli', 'snli', 'mnli', 'anli.round1', 'anli.round2', 'anli.round3')
analysistype2tnendings = {
    'pos': ['mask_noun', 'mask_verb', 'mask_adj', 'mask_adv', 'mask_prep', 'mask_noun_fraction', 'mask_verb_fraction', 'mask_adj_fraction', 'mask_adv_fraction', 'mask_prep_fraction'],
    'gender': ['mask_male_words', 'mask_female_words', 'mask_male_words_fraction', 'mask_female_words_fraction'],
    'causal': ['mask_causal_words', 'mask_causal_words_fraction'],
    'logical': ['mask_logical_words', 'mask_logical_words_fraction'],
    'content': ['mask_content_pos_words', 'mask_content_pos_words_fraction'],
    'male': ['mask_male_words', 'mask_male_words_fraction'],
    'female': ['mask_female_words', 'mask_female_words_fraction'],
    'shuffle': ['shuffle'],
    'length': ['length'],
    'mask_all': ['mask_all'],
}
for task in glue_nli_tasks:
    for analysis_type, tn_endings in analysistype2tnendings.items():
        group2datasethps[f'{task}.{analysis_type}'] = {'mn': all_models, 'tn': [get_taskname(task, tn_ending) for tn_ending in [''] + tn_endings]}
