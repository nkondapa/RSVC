import matplotlib.pyplot as plt
import numpy as np
import torch
from math import ceil

from src.utils import saving, model_loader, concept_extraction_helper as ceh
import json
import os
from tqdm import tqdm

from extract_model_activations import create_image_group, _batch_inference
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from concept_integrated_gradients import build_output_dir as build_importance_output_dir
from compare_models import build_model_comparison_parser, set_seed, build_output_dir, process_config
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib
import seaborn as sns
from src.utils import plotting_helper as ph
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages


def convert_to_array(statistic_dict):
    metadata = statistic_dict['metadata']
    del statistic_dict['metadata']
    arr = np.zeros((metadata['num_concepts_i'], metadata['num_concepts_j']))
    for i in range(metadata['num_concepts_i']):
        for j in range(metadata['num_concepts_j']):
            key = (i, j)
            value = statistic_dict['1to2'][key]
            arr[key[0], key[1]] = value.statistic

    return arr


def process_comparison_dict(method, comparison_dir, class_idx, m0_layer, m1_layer, patchify=False, baseline=False, use_train_data=False):
    '''

    :param method:
    :param comparison_dict:
    :return: max_concept_match_rtoc, max_concept_match_ctor
    max_concept_match_rtoc:
        correlation: given a concept in model1 what is the best match from model2  (max in each row)
        regression: how well does model2 predict a model1 concept
    max_concept_match_ctor:
        correlation: given a concept in model2 what is the best match from model1  (max in each column)
        regression: how well does model1 predict a model2 concept
    '''

    def load_comp_dict():
        p = os.path.join(comparison_dir, f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl')
        try:
            with open(p, 'rb') as f:
                comparison_dict = pkl.load(f)
        except:
            comparison_dict = None
        return comparison_dict

    def load_regression_eval_dict(patched=False):
        folder_name = 'regression_evaluation' if not patched else 'regression_evaluation_patched'
        eval_dir = comparison_dir.replace('concept_comparison', folder_name)
        p = os.path.join(eval_dir, f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl')
        try:
            with open(p, 'rb') as f:
                eval_dict = pkl.load(f)
        except:
            eval_dict = None
        return eval_dict

    if method == 'pearson':
        comparison_dict = load_comp_dict()
        _arr = convert_to_array(comparison_dict)
        _arr[np.isnan(_arr)] = 0
        max_concept_match_rtoc = np.nanmax(_arr, axis=1)
        max_concept_match_ctor = np.nanmax(_arr, axis=0)
    elif method == 'spearman':
        comparison_dict = load_comp_dict()
        _arr = convert_to_array(comparison_dict)
        _arr[np.isnan(_arr)] = 0
        max_concept_match_rtoc = np.nanmax(_arr, axis=1)
        max_concept_match_ctor = np.nanmax(_arr, axis=0)
    elif method in ['lasso_regression', 'lasso_regression_c', 'k_nearest_neighbors'] and not use_train_data:
        # print('Loading regression evaluation dict')
        eval_dict = load_regression_eval_dict(patchify)
        if eval_dict is None:
            return dict(max_concept_match_rtoc=None, max_concept_match_ctor=None)
        if baseline:
            max_concept_match_ctor = eval_dict['scores']['pearsonr']['2to2']
            max_concept_match_rtoc = eval_dict['scores']['pearsonr']['1to1']
        else:
            max_concept_match_ctor = eval_dict['scores']['pearsonr']['1to2']
            max_concept_match_rtoc = eval_dict['scores']['pearsonr']['2to1']
    elif method in ['lasso_regression', 'lasso_regression_c', 'k_nearest_neighbors', 'concept_lasso_regression',
                    'concept_lasso_regression_c']:
        comparison_dict = load_comp_dict()
        if comparison_dict is None:
            return dict(max_concept_match_rtoc=None, max_concept_match_ctor=None)
        if 'lr1to2_pearson' in comparison_dict: # old version no kfold
            max_concept_match_ctor = comparison_dict['lr1to2_pearson']
            max_concept_match_rtoc = comparison_dict['lr2to1_pearson']
        else:
            # compute mean max_concept_match_ctor
            lr1to2_pearson = []
            lr2to1_pearson = []
            for k in comparison_dict.keys():
                lr2to1_pearson.append(comparison_dict[k]['lr2to1_pearson'])
                lr1to2_pearson.append(comparison_dict[k]['lr1to2_pearson'])

            lr1to2_pearson = np.array(lr1to2_pearson)
            lr2to1_pearson = np.array(lr2to1_pearson)

            lr1to2_pearson_mean = np.nanmean(lr1to2_pearson, axis=0)
            lr2to1_pearson_mean = np.nanmean(lr2to1_pearson, axis=0)
            max_concept_match_ctor = lr1to2_pearson_mean
            max_concept_match_rtoc = lr2to1_pearson_mean

            lr1to2_pearson_std = np.nanstd(lr1to2_pearson, axis=0)
            lr2to1_pearson_std = np.nanstd(lr2to1_pearson, axis=0)
    else:
        print()
        raise ValueError(f'Unknown method {method}')

    return dict(max_concept_match_rtoc=max_concept_match_rtoc, max_concept_match_ctor=max_concept_match_ctor)


def visualize_similarity_vs_importance(class_list, comparison_dir, fe_outs, eval_dict,
                                       int_grad0, int_grad1, plot_params=None):
    show = plot_params.get('show', False)
    save = plot_params.get('save', False)
    visualization_output_dir = plot_params.get('visualization_output_dir', None)
    model0_name = plot_params.get('model0_name', None)
    model1_name = plot_params.get('model1_name', None)
    model0_plot_name = plot_params.get('model0_plot_name', None)
    model1_plot_name = plot_params.get('model1_plot_name', None)
    visualize_each_class = plot_params.get('visualize_each_class', False)
    visualize_summary_plot_comparison = plot_params.get('visualize_summary_plot_comparison', False)
    method = plot_params['method']
    use_train_data = plot_params.get('use_train_data', False)
    class_specific_viz_folder_model0 = None
    class_specific_viz_folder_model1 = None
    summary_plot_comparison_dir = None
    patchify = plot_params.get('patchify', False)
    baseline = plot_params.get('baseline', False)
    overlay_mean = plot_params.get('overlay_mean', False)
    compute_weighted_int_grad = False if patchify or (eval_dict is None) else True
    fontsize = plot_params.get('fontsize', 16)

    def _set_ax_color(ax, color):
        # Set the color of the x and y axes to red
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)

        # Optionally, you can also set the color of the top and right spines
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)

    if visualize_each_class:
        class_specific_viz_folder_model0 = os.path.join(visualization_output_dir, 'class_similarity_vs_importance',
                                                        model0_name)
        class_specific_viz_folder_model1 = os.path.join(visualization_output_dir, 'class_similarity_vs_importance',
                                                        model1_name)
        os.makedirs(class_specific_viz_folder_model0, exist_ok=True)
        os.makedirs(class_specific_viz_folder_model1, exist_ok=True)
    if visualize_summary_plot_comparison:
        summary_plot_comparison_dir = os.path.join(visualization_output_dir, 'summary_plot_comparison')
        os.makedirs(summary_plot_comparison_dir, exist_ok=True)

    weighted_int_grad0 = []
    weighted_int_grad1 = []
    m0_f1 = []
    m1_f1 = []
    max_wig = None
    min_wig = None
    for ci in range(len(class_list)):
        class_idx = class_list[ci]

        if compute_weighted_int_grad:
            prob_delta = eval_dict[model0_name]['sample'][class_idx] - eval_dict[model1_name]['sample'][class_idx]
            m0_f1.append(eval_dict[model0_name]['stats'][str(class_list[ci])]['f1'])
            m1_f1.append(eval_dict[model1_name]['stats'][str(class_list[ci])]['f1'])
            prob_delta = torch.FloatTensor(prob_delta).unsqueeze(1)
            weighted_int_grad0.append(int_grad0[ci] * prob_delta)
            weighted_int_grad1.append(int_grad1[ci] * prob_delta * -1)
            _max = max(weighted_int_grad0[ci].mean(0).max(), weighted_int_grad1[ci].mean(0).max())
            _min = min(weighted_int_grad0[ci].mean(0).min(), weighted_int_grad1[ci].mean(0).min())
            if max_wig is None or _max > max_wig:
                max_wig = _max
            if min_wig is None or _min < min_wig:
                min_wig = _min
    if compute_weighted_int_grad:
        m0_f1 = np.array(m0_f1)
        m1_f1 = np.array(m1_f1)

    pbar = tqdm(class_list)
    m0_layer = fe_outs[0]['layer_names'][-1]
    m1_layer = fe_outs[1]['layer_names'][-1]

    ci = 0
    rtoc_list = []
    ctor_list = []
    int_grad0_list = []
    int_grad1_list = []
    wig0_list = []
    wig1_list = []
    valid_class_indices = []
    for class_idx in pbar:

        out = process_comparison_dict(method, comparison_dir, class_idx, m0_layer, m1_layer, patchify=patchify, baseline=baseline, use_train_data=use_train_data)
        max_concept_match_rtoc, max_concept_match_ctor = out['max_concept_match_rtoc'], out['max_concept_match_ctor']
        if max_concept_match_rtoc is None or max_concept_match_ctor is None:
            print(f'Class {class_idx} not found')
            continue
        valid_class_indices.append(class_idx)
        rtoc_list.extend(list(max_concept_match_rtoc))
        ctor_list.extend(list(max_concept_match_ctor))
        int_grad0_list.extend(list(int_grad0[ci].mean(0)))
        int_grad1_list.extend(list(int_grad1[ci].mean(0)))
        if compute_weighted_int_grad:
            wig0_list.extend(list(weighted_int_grad0[ci].mean(0)))
            wig1_list.extend(list(weighted_int_grad1[ci].mean(0)))

        # ax.set_ylim([0, 10])
        # ax.set_yscale('log')

        if visualize_each_class:
            fig_r, axes_r = plt.subplots(2, 1)
            axes_r[0].scatter(max_concept_match_rtoc, int_grad0[ci].mean(0), alpha=0.5, color='blue')
            [axes_r[0].annotate(f'{i}', (max_concept_match_rtoc[i], int_grad0[ci].mean(0)[i])) for i in
             range(len(max_concept_match_ctor))]
            axes_r[1].scatter(max_concept_match_rtoc, weighted_int_grad0[ci].mean(0), alpha=0.5, color='blue')
            [axes_r[1].annotate(f'{i}', (max_concept_match_rtoc[i], weighted_int_grad0[ci].mean(0)[i])) for i in
             range(len(max_concept_match_ctor))]
            axes_r[0].set_xlim([-0.2, 1])
            f1, acc = eval_dict[model0_name]['stats'][str(class_idx)]['f1'], \
            eval_dict[model0_name]['stats'][str(class_idx)]['acc']
            plt.suptitle(f'{class_idx} | {model0_name} | F1: {f1:0.2f} | Acc: {acc:0.2f}')
            plt.savefig(os.path.join(class_specific_viz_folder_model0, f'{class_idx}.png'))
            plt.close(fig_r)

            fig_c, axes_c = plt.subplots(2, 1)
            axes_c[0].scatter(max_concept_match_ctor, int_grad1[ci].mean(0), alpha=0.5, color='blue')
            [axes_c[0].annotate(f'{i}', (max_concept_match_ctor[i], int_grad1[ci].mean(0)[i])) for i in
             range(len(max_concept_match_ctor))]
            axes_c[1].scatter(max_concept_match_ctor, weighted_int_grad1[ci].mean(0), alpha=0.5, color='blue')
            [axes_c[1].annotate(f'{i}', (max_concept_match_ctor[i], weighted_int_grad1[ci].mean(0)[i])) for i in
             range(len(max_concept_match_ctor))]
            axes_c[0].set_xlim([-0.2, 1])
            f1, acc = eval_dict[model1_name]['stats'][str(class_idx)]['f1'], \
            eval_dict[model1_name]['stats'][str(class_idx)]['acc']
            plt.suptitle(f'{class_idx} | {model1_name} | F1: {f1:0.2f} | Acc: {acc:0.2f}')
            plt.savefig(os.path.join(class_specific_viz_folder_model1, f'{class_idx}.png'))
            plt.close(fig_c)

        if visualize_summary_plot_comparison:
            concept_viz_dir1 = plot_params['concept_viz_dir1']
            concept_viz_dir2 = plot_params['concept_viz_dir2']
            root = '/media/nkondapa/SSD2/concept_book/'
            concept_viz_dir1 = concept_viz_dir1.replace('./', root)
            concept_viz_dir2 = concept_viz_dir2.replace('./', root)
            wig0 = weighted_int_grad0[ci].mean(0)
            wig1 = weighted_int_grad1[ci].mean(0)
            cmin_wig = min(wig0.min(), wig1.min())
            cmax_wig = max(wig0.max(), wig1.max())
            nwig0 = (wig0 - wig0.min()) / (wig0.max() - wig0.min())
            nwig1 = (wig1 - wig1.min()) / (wig1.max() - wig1.min())

            ig0 = int_grad0[ci].mean(0)
            ig1 = int_grad1[ci].mean(0)
            sim0 = max_concept_match_rtoc
            sim1 = max_concept_match_ctor

            img_m1 = [Image.open(
                os.path.join(concept_viz_dir1, m0_layer, f'{class_idx}', 'top10/summary_plots', f'concept_{i}.png')) for
                      i in range(10)]
            img_m2 = [Image.open(
                os.path.join(concept_viz_dir2, m1_layer, f'{class_idx}', 'top10/summary_plots', f'concept_{i}.png')) for
                      i in range(10)]

            sum_fig, sum_axes = plt.subplots(10, 2)
            sum_fig.set_size_inches(12, 24)
            sorted_ind0 = torch.argsort(torch.FloatTensor(sim0), descending=False)
            sorted_ind1 = torch.argsort(torch.FloatTensor(sim1), descending=False)
            for i in range(10):
                ind0 = sorted_ind0[i]
                ind1 = sorted_ind1[i]
                sum_axes[i, 0].imshow(img_m1[ind0])
                sum_axes[i, 1].imshow(img_m2[ind1])
                sum_axes[i, 0].set_xlabel(f'Sim: {sim0[ind0]:0.2f} | IG: {ig0[ind0]:0.2f} | WIG: {wig0[ind0]:0.2f}')
                sum_axes[i, 1].set_xlabel(f'Sim: {sim1[ind1]:0.2f} | IG: {ig1[ind1]:0.2f} | WIG: {wig1[ind1]:0.2f}')
                sum_axes[i, 0].set_xticks([])
                sum_axes[i, 1].set_xticks([])
                sum_axes[i, 0].set_yticks([])
                sum_axes[i, 1].set_yticks([])
                # _set_ax_color(sum_axes[i, 0], col0)
                # _set_ax_color(sum_axes[i, 1], col1)
            m0_f1 = eval_dict[model0_name]['stats'][str(class_idx)]['f1']
            m0_acc = eval_dict[model0_name]['stats'][str(class_idx)]['acc']
            m1_f1 = eval_dict[model1_name]['stats'][str(class_idx)]['f1']
            m1_acc = eval_dict[model1_name]['stats'][str(class_idx)]['acc']
            sum_axes[0, 0].set_title(f'{class_idx} | {model0_name} | F1: {m0_f1:0.2f} | Acc: {m0_acc:0.2f}')
            sum_axes[0, 1].set_title(f'{class_idx} | {model1_name} | F1: {m1_f1:0.2f} | Acc: {m1_acc:0.2f}')
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(summary_plot_comparison_dir, f'{class_idx}.png'))
            plt.close(sum_fig)
        ci += 1

    ig0_min, ig0_max = min(int_grad0_list).item(), max(int_grad0_list).item()
    ig1_min, ig1_max = min(int_grad1_list).item(), max(int_grad1_list).item()

    rtoc_arr = np.array(rtoc_list)
    ctor_arr = np.array(ctor_list)

    num_birds = (np.array(valid_class_indices) < 555).sum()
    print(rtoc_arr[num_birds * 10:].mean(), ctor_arr[num_birds * 10:].mean())
    print(rtoc_arr[:num_birds * 10].mean(), ctor_arr[:num_birds * 10].mean())

    rtoc_mask = ~np.isnan(rtoc_arr)
    ctor_mask = ~np.isnan(ctor_arr)

    rtoc_arr = rtoc_arr[rtoc_mask]
    ctor_arr = ctor_arr[ctor_mask]

    ig0_arr = np.array(int_grad0_list)[rtoc_mask]
    ig1_arr = np.array(int_grad1_list)[ctor_mask]

    if compute_weighted_int_grad:
        wig0_arr = np.array(wig0_list)[rtoc_mask]
        wig1_arr = np.array(wig1_list)[ctor_mask]

    # rtoc_arr[np.isnan(rtoc_arr)] = 0
    # ctor_arr[np.isnan(ctor_arr)] = 0

    min_ig = min(ig0_min, ig1_min)
    max_ig = max(ig0_max, ig1_max)
    # TODO normalize importance 0 to 1
    # ig0_arr = (ig0_arr - ig0_arr.min()) / (ig0_arr.max() - ig0_arr.min())
    # ig1_arr = (ig1_arr - ig0_arr.min()) / (ig0_arr.max() - ig0_arr.min())
    print(np.abs(ig0_arr).max(), np.abs(ig1_arr).max())
    imp0 = ig0_arr / np.abs(ig0_arr).max()
    imp1 = ig1_arr / np.abs(ig1_arr).max()

    fig, axes = plt.subplots(2, 1, squeeze=False)
    fig.set_size_inches(6, 6)
    ax = axes[0, 0]
    ax.hist(rtoc_arr, bins=np.linspace(-0.2, 1, 10), alpha=0.5, color='blue', label='rtoc')
    ax.set_xlabel(f'{model1_plot_name}' + r'$\rightarrow$' + f'{model0_plot_name} Concept Similarity')
    ax = axes[1, 0]
    ax.hist(ctor_arr, bins=np.linspace(-0.2, 1, 10), alpha=0.5, color='red', label='ctor')
    ax.set_xlabel(f'{model1_plot_name}' + r'$\rightarrow$' + f'{model0_plot_name} Concept Similarity')
    if save:
        plt.savefig(os.path.join(visualization_output_dir, 'similarity_histogram.png'))
    if show:
        plt.show()

    fig, axes = plt.subplots(2, 1, squeeze=False, constrained_layout=True)
    fig.set_size_inches(6, 4)
    z0 = gaussian_kde(np.vstack([rtoc_arr, imp0]))(np.vstack([rtoc_arr, imp0]))
    z1 = gaussian_kde(np.vstack([ctor_arr, imp1]))(np.vstack([ctor_arr, imp1]))
    cmax = max(z0.max(), z1.max())
    xmin = min(rtoc_arr.min(), ctor_arr.min())

    ax = axes[0, 0]
    si = np.argsort(z0)
    im = ax.scatter(rtoc_arr[si], imp0[si], alpha=0.3, c=z0[si], cmap='magma', vmin=0, vmax=cmax)
    if baseline:
        # ax.set_xlabel(f'{model0_plot_name} $\\rightarrow$ {model0_plot_name} Concept Similarity', fontsize=fontsize)
        ax.set_xlabel(f'{model0_plot_name} $\\rightarrow$ {model0_plot_name}', fontsize=fontsize)
    else:
        # ax.set_xlabel(f'{model1_plot_name} $\\rightarrow$ {model0_plot_name} Concept Similarity', fontsize=fontsize)
        ax.set_xlabel(f'{model1_plot_name} $\\rightarrow$ {model0_plot_name}', fontsize=fontsize)
    # ax.set_ylabel(f'{model0_plot_name} CI', fontsize=fontsize)
    ax.set_ylabel(f'{model0_plot_name}', fontsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.4, 1.05])
    if overlay_mean:
        ax.axvline(x=rtoc_arr.mean(), color='black', linestyle='-', linewidth=3)

    ax = axes[1, 0]
    si = np.argsort(z1)
    im = ax.scatter(ctor_arr[si], imp1[si], alpha=0.3, c=z1[si], cmap='magma', vmin=0, vmax=cmax)
    if baseline:
        # ax.set_xlabel(f'{model1_plot_name} $\\rightarrow$ {model1_plot_name} Concept Similarity', fontsize=fontsize)
        ax.set_xlabel(f'{model1_plot_name} $\\rightarrow$ {model1_plot_name}', fontsize=fontsize)
    else:
        # ax.set_xlabel(f'{model0_plot_name} $\\rightarrow$ {model1_plot_name} Concept Similarity', fontsize=fontsize)
        ax.set_xlabel(f'{model0_plot_name} $\\rightarrow$ {model1_plot_name}', fontsize=fontsize)
    # ax.set_ylabel(f'{model1_plot_name} CI', fontsize=fontsize)
    ax.set_ylabel(f'{model1_plot_name}', fontsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.4, 1.05])
    if overlay_mean:
        ax.axvline(x=ctor_arr.mean(), color='black', linestyle='-', linewidth=3)
    # ax.set_ylim([min_ig - 0.1, max_ig + 0.1])
    # plt.tight_layout()
    # plt.show()
    interval = ceil(cmax / 5)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=range(0, ceil(cmax), interval))
    cbar.solids.set(alpha=1)
    # plt.show()

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    stats = {'mean_similarity': {'2to1': rtoc_arr.mean(), '1to2': ctor_arr.mean()},
             'median_similarity': {'2to1': np.median(rtoc_arr), '1to2': np.median(ctor_arr)},
             'std_similarity': {'2to1': rtoc_arr.std(), '1to2': ctor_arr.std()},
             'weighted_mean_similarity': {'2to1': np.average(rtoc_arr, weights=ig0_arr),
                                          '1to2': np.average(ctor_arr, weights=ig1_arr)},
             'weighted_median_similarity': {'2to1': np.median(rtoc_arr * ig0_arr), '1to2': np.median(ctor_arr * ig1_arr)},
             }
    with open(os.path.join(visualization_output_dir, 'similarity_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    if save:
        plt.savefig(os.path.join(visualization_output_dir, 'similarity_vs_importance.png'), dpi=600)
        plt.savefig(os.path.join(visualization_output_dir, 'similarity_vs_importance.pdf'))
    if show:
        plt.show()

    fig, axes = plt.subplots(2, 1, squeeze=False, constrained_layout=True)
    fig.set_size_inches(6, 5)

    # wig1_arr = (wig0_arr - wig0_arr.mean()) / wig0_arr.std()
    # wig1_arr = (wig1_arr - wig1_arr.mean()) / wig1_arr.std()
    if compute_weighted_int_grad:
        imp0 = wig0_arr / np.abs(wig0_arr).max()
        imp1 = wig1_arr / np.abs(wig1_arr).max()

        z0 = gaussian_kde(np.vstack([rtoc_arr, imp0]))(np.vstack([rtoc_arr, imp0]))
        z1 = gaussian_kde(np.vstack([ctor_arr, imp1]))(np.vstack([ctor_arr, imp1]))
        cmax = max(z0.max(), z1.max())

        ax = axes[0, 0]
        si = np.argsort(z0)
        ax.scatter(rtoc_arr[si], imp0[si], alpha=0.3, c=z0[si], cmap='magma', vmin=0, vmax=cmax)
        ax.set_xlabel(f'{model1_plot_name} to {model0_plot_name} Concept Similarity')
        ax.set_ylabel(f'{model0_plot_name} WCI')
        # ax.set_xlim([-0.2, 1])
        # ax.set_ylim([min_wig - 0.1, max_wig + 0.1])

        ax = axes[1, 0]
        si = np.argsort(z1)
        ax.scatter(ctor_arr[si], imp1[si], alpha=0.3, c=z1[si], cmap='magma', vmin=0, vmax=cmax)
        ax.set_xlabel(f'{model0_plot_name} to {model1_plot_name} Concept Similarity')
        ax.set_ylabel(f'{model1_plot_name} WCI')
        # ax.set_xlim([-0.2, 1])
        # ax.set_ylim([min_wig - 0.1, max_wig + 0.1])
        # plt.tight_layout()
        # plt.show()

        if save:
            plt.savefig(os.path.join(visualization_output_dir, 'similarity_vs_weighted_importance.png'))
        if show:
            plt.show()

    plt.close()

    index_to_concept_id = {}
    num_concepts = int_grad1[0].shape[1]
    ti = 0
    for ci, class_idx in enumerate(valid_class_indices):
        for ki in range(num_concepts):
            index_to_concept_id[ti] = (class_idx, ki)
            ti += 1

    imp0 = ig0_arr / np.abs(ig0_arr).max()
    imp1 = ig1_arr / np.abs(ig1_arr).max()
    dist0 = np.linalg.norm(np.array([[0], [1]]) - np.stack([rtoc_arr, imp0], axis=0), axis=0)
    dist1 = np.linalg.norm(np.array([[0], [1]]) - np.stack([ctor_arr, imp1], axis=0), axis=0)

    # distance to sim 0, importance 1
    # output_name = 'low_sim_high_imp'
    # samples0 = []
    # for ind in dist0.argsort()[:20]:
    #     class_idx, concept_idx = index_to_concept_id[ind]
    #     samples0.append({'class': class_idx, 'concept': concept_idx, 'similarity': rtoc_arr[ind].item(), 'ci': ig0_arr[ind].item()})#, 'wci': wig0_arr[ind].item()})
    #
    # samples1 = []
    # for ind in dist1.argsort()[:20]:
    #     class_idx, concept_idx = index_to_concept_id[ind]
    #     samples1.append({'class': class_idx, 'concept': concept_idx, 'similarity': ctor_arr[ind].item(), 'ci': ig1_arr[ind].item()})#, 'wci': wig1_arr[ind].item()})

    output_name = 'low_sim_imp_gt_thresh=0.5'
    i0_thresh = np.sort(imp0)[int(0.5 * len(imp0))]
    i1_thresh = np.sort(imp1)[int(0.5 * len(imp0))]
    mask0 = imp0 > i0_thresh
    mask1 = imp1 > i1_thresh
    tmp = np.arange(len(rtoc_list))[rtoc_mask]
    sel_inds0 = tmp[mask0][rtoc_arr[mask0].argsort()]
    tmp = np.arange(len(rtoc_list))[ctor_mask]
    sel_inds1 = tmp[mask1][ctor_arr[mask1].argsort()]
    samples0 = []
    for ind in sel_inds0[:20]:
        class_idx, concept_idx = index_to_concept_id[ind]
        samples0.append({'class': class_idx, 'concept': concept_idx, 'similarity': rtoc_arr[ind].item(), 'ci': ig0_arr[ind].item()})#, 'wci': wig0_arr[ind].item()})

    samples1 = []
    for ind in sel_inds1[:20]:
        class_idx, concept_idx = index_to_concept_id[ind]
        samples1.append({'class': class_idx, 'concept': concept_idx, 'similarity': ctor_arr[ind].item(), 'ci': ig1_arr[ind].item()})#, 'wci': wig1_arr[ind].item()})

    # output_name = 'low_sim'
    # samples0 = []
    # for ind in rtoc_arr.argsort()[:20]:
    #     class_idx, concept_idx = index_to_concept_id[ind]
    #     samples0.append(
    #         {'class': class_idx, 'concept': concept_idx, 'similarity': rtoc_arr[ind].item(), 'ci': ig0_arr[ind].item()})
    #
    # samples1 = []
    # for ind in ctor_arr.argsort()[:20]:
    #     class_idx, concept_idx = index_to_concept_id[ind]
    #     samples1.append(
    #         {'class': class_idx, 'concept': concept_idx, 'similarity': ctor_arr[ind].item(), 'ci': ig1_arr[ind].item()})

    sample_class_dict = {}
    for sample in samples0:
        sample_class_dict[sample['class']] = -1
    for sample in samples1:
        if sample['class'] in sample_class_dict:
            sample_class_dict[sample['class']] = 0
        else:
            sample_class_dict[sample['class']] = 1

    print()

    # samples0 = []
    # samples1 = []
    # bins = np.linspace(-0.2, 1, 10)
    # for i in range(len(bins) - 1):
    #     mask = (rtoc_arr >= bins[i]) & (rtoc_arr < bins[i + 1])
    #     indices = np.where(mask)[0]
    #     if len(indices) != 0:
    #         for imp in [ig0_arr, wig0_arr]:
    #             _max_ind = imp[indices].argmax()
    #             max_ind = indices[_max_ind]
    #             class_idx, concept_idx = index_to_concept_id[max_ind]
    #             samples0.append({'class': class_idx, 'concept': concept_idx, 'similarity': rtoc_arr[max_ind].item(), 'ci': ig0_arr[max_ind].item(), 'wci': wig0_arr[max_ind].item()})
    #
    #     mask = (ctor_arr >= bins[i]) & (ctor_arr < bins[i + 1])
    #     indices = np.where(mask)[0]
    #     if len(indices) != 0:
    #         for imp in [ig1_arr, wig1_arr]:
    #             _max_ind = imp[indices].argmax()
    #             max_ind = indices[_max_ind]
    #             class_idx, concept_idx = index_to_concept_id[max_ind]
    #             samples1.append({'class': class_idx, 'concept': concept_idx, 'similarity': ctor_arr[max_ind].item(), 'ci': ig1_arr[max_ind].item(), 'wci': wig1_arr[max_ind].item()})
    #
    # samples0 = []
    # samples1 = []
    # bins = np.linspace(-0.2, 1, 10)
    # for i in range(len(bins) - 1):
    #     mask = (rtoc_arr >= bins[i]) & (rtoc_arr < bins[i + 1])
    #     indices = np.where(mask)[0]
    #     if len(indices) != 0:
    #         for imp in [ig0_arr, wig0_arr]:
    #             _max_ind = imp[indices].argmax()
    #             max_ind = indices[_max_ind]
    #             class_idx, concept_idx = index_to_concept_id[max_ind]
    #             samples0.append({'class': class_idx, 'concept': concept_idx, 'similarity': rtoc_arr[max_ind].item(), 'ci': ig0_arr[max_ind].item(), 'wci': wig0_arr[max_ind].item()})
    #
    #     mask = (ctor_arr >= bins[i]) & (ctor_arr < bins[i + 1])
    #     indices = np.where(mask)[0]
    #     if len(indices) != 0:
    #         for imp in [ig1_arr, wig1_arr]:
    #             _max_ind = imp[indices].argmax()
    #             max_ind = indices[_max_ind]
    #             class_idx, concept_idx = index_to_concept_id[max_ind]
    #             samples1.append({'class': class_idx, 'concept': concept_idx, 'similarity': ctor_arr[max_ind].item(), 'ci': ig1_arr[max_ind].item(), 'wci': wig1_arr[max_ind].item()})
    #
    s0_class_list = list(set([x['class'] for x in samples0]))
    s1_class_list = list(set([x['class'] for x in samples1]))

    s0_class_list = np.unique(s0_class_list).tolist()
    s1_class_list = np.unique(s1_class_list).tolist()

    with open(os.path.join(visualization_output_dir, f'{output_name}_samples0_class_list.json'), 'w') as f:
        json.dump(s0_class_list, f, indent=2)
    with open(os.path.join(visualization_output_dir, f'{output_name}_samples1_class_list.json'), 'w') as f:
        json.dump(s1_class_list, f, indent=2)

    with open(os.path.join(visualization_output_dir, f'{output_name}_samples0.json'), 'w') as f:
        json.dump(samples0, f, indent=2)
    with open(os.path.join(visualization_output_dir, f'{output_name}_samples1.json'), 'w') as f:
        json.dump(samples1, f, indent=2)

    class_list_dict_path = os.path.join('./', f'{output_name}')
    os.makedirs(class_list_dict_path, exist_ok=True)
    comp_name = comparison_dir.split('concept_comparison/')[-1].split('/')[0]
    with open(os.path.join(class_list_dict_path, f'{comp_name}.json'), 'w') as f:
        json.dump(sample_class_dict, f, indent=2)

    return dict(samples_class_lists=[s0_class_list, s1_class_list], rtoc=rtoc_arr, ctor=ctor_arr,
                ig0=int_grad0, ig1=int_grad1, wig0=weighted_int_grad0, wig1=weighted_int_grad1, rtoc_arr=rtoc_arr,
                ctor_arr=ctor_arr,
                m0_layer=m0_layer, m1_layer=m1_layer)


def visualize_similarity_samples(samples_class_lists, eval_dict, plot_params=None):
    show = plot_params.get('show', False)
    save = plot_params.get('save', False)
    class_list = plot_params['class_list']
    visualization_output_dir = plot_params.get('visualization_output_dir', None)
    model0_name = plot_params.get('model0_name', None)
    model1_name = plot_params.get('model1_name', None)
    summary_plot_comparison_dir = os.path.join(visualization_output_dir, 'summary_plot_comparison')
    os.makedirs(summary_plot_comparison_dir, exist_ok=True)

    ig0_list = plot_params['ig0']
    ig1_list = plot_params['ig1']
    wig0_list = plot_params['wig0']
    wig1_list = plot_params['wig1']
    ig0_arr = np.stack([x.mean(0) for x in ig0_list])
    ig1_arr = np.stack([x.mean(0) for x in ig1_list])
    wig0_arr = np.stack([x.mean(0) for x in wig0_list])
    wig1_arr = np.stack([x.mean(0) for x in wig1_list])
    rtoc_arr = plot_params['rtoc_arr'].reshape(ig0_arr.shape[0], -1)
    ctor_arr = plot_params['ctor_arr'].reshape(ig1_arr.shape[0], -1)
    m0_layer = plot_params['m0_layer']
    m1_layer = plot_params['m1_layer']

    def _set_ax_color(ax, color):
        # Set the color of the x and y axes to red
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)

        # Optionally, you can also set the color of the top and right spines
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)

    s0_class_list, s1_class_list = samples_class_lists
    sample_list = s0_class_list + s1_class_list
    for i in range(2):
        pdf = PdfPages(os.path.join(visualization_output_dir, f'samples{i}.pdf'))
        for class_idx in tqdm(sample_list):
            ci = class_list.index(class_idx)
            concept_viz_dir1 = plot_params['concept_viz_dir1']
            concept_viz_dir2 = plot_params['concept_viz_dir2']
            root = '/media/nkondapa/SSD2/concept_book/'
            concept_viz_dir1 = concept_viz_dir1.replace('./', root)
            concept_viz_dir2 = concept_viz_dir2.replace('./', root)
            wig0 = wig0_arr[ci]
            wig1 = wig1_arr[ci]
            nwig0 = wig0 / np.abs(wig0_arr).max()
            nwig1 = wig1 / np.abs(wig1_arr).max()

            ig0 = ig0_arr[ci]
            ig1 = ig1_arr[ci]
            norm_ig0 = ig0 / np.abs(ig0_arr).max()
            norm_ig1 = ig1 / np.abs(ig1_arr).max()

            sim0 = rtoc_arr[ci]
            sim1 = ctor_arr[ci]

            img_m1 = [Image.open(
                os.path.join(concept_viz_dir1, m0_layer, f'{class_idx}', 'top10/summary_plots', f'concept_{i}.png')) for
                      i in range(10)]
            img_m2 = [Image.open(
                os.path.join(concept_viz_dir2, m1_layer, f'{class_idx}', 'top10/summary_plots', f'concept_{i}.png')) for
                      i in range(10)]

            sum_fig, sum_axes = plt.subplots(10, 2)
            sum_fig.set_size_inches(12, 24)
            sorted_ind0 = torch.argsort(torch.FloatTensor(sim0), descending=False)
            sorted_ind1 = torch.argsort(torch.FloatTensor(sim1), descending=False)

            for i in range(10):
                ind0 = sorted_ind0[i]
                ind1 = sorted_ind1[i]
                sum_axes[i, 0].imshow(img_m1[ind0])
                sum_axes[i, 1].imshow(img_m2[ind1])
                sum_axes[i, 0].set_xlabel(
                    f'Sim: {sim0[ind0]:0.2f} | IG: {norm_ig0[ind0]:0.2f} | WIG: {nwig0[ind0]:0.2f}')
                sum_axes[i, 1].set_xlabel(
                    f'Sim: {sim1[ind1]:0.2f} | IG: {norm_ig1[ind1]:0.2f} | WIG: {nwig1[ind1]:0.2f}')
                sum_axes[i, 0].set_xticks([])
                sum_axes[i, 1].set_xticks([])
                sum_axes[i, 0].set_yticks([])
                sum_axes[i, 1].set_yticks([])
                # _set_ax_color(sum_axes[i, 0], col0)
                # _set_ax_color(sum_axes[i, 1], col1)
            m0_f1 = eval_dict[model0_name]['stats'][str(class_idx)]['f1']
            m0_acc = eval_dict[model0_name]['stats'][str(class_idx)]['acc']
            m1_f1 = eval_dict[model1_name]['stats'][str(class_idx)]['f1']
            m1_acc = eval_dict[model1_name]['stats'][str(class_idx)]['acc']
            sum_axes[0, 0].set_title(f'{class_idx} | {model0_name} | F1: {m0_f1:0.2f} | Acc: {m0_acc:0.2f}')
            sum_axes[0, 1].set_title(f'{class_idx} | {model1_name} | F1: {m1_f1:0.2f} | Acc: {m1_acc:0.2f}')
            plt.tight_layout()
            # plt.show()
            pdf.savefig(sum_fig)
            plt.savefig(os.path.join(summary_plot_comparison_dir, f'{class_idx}.png'))
            plt.close(sum_fig)

        pdf.close()


def load_model_stats(args, dataset_name, image_group, param_dicts1, param_dicts2, model_save_names):
    model_eval = {}
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_save_name = model_save_names[mi]
        path = f"model_evaluation/{dataset_name}/{model_name}_probs_{args.data_split}.pth"
        print(path)
        probs = torch.load(path)
        with open(f'model_evaluation/{dataset_name}/{model_name}_{args.data_split}.json', 'r') as f:
            eval_dict = json.load(f)
        with open(f'model_evaluation/{dataset_name}/{model_name}_stats_{args.data_split}.json', 'r') as f:
            stats = json.load(f)

        print(f"Model: {model_save_name}")
        img_paths = np.array(list(eval_dict['predictions'].keys()))
        labels = np.array(eval_dict['labels'])
        model_eval[model_save_name] = {'mean': [], 'sample': [], 'stats': stats}
        for class_idx in np.unique(labels):
            # mask = labels == class_idx
            # mask = np.array([path in sampled_paths for path in img_paths])
            # class_paths = img_paths[mask]
            class_paths = image_group[class_idx]
            class_prob = np.array([probs[path][class_idx] for path in class_paths])
            model_eval[model_save_name]['mean'].append(class_prob.mean())
            model_eval[model_save_name]['sample'].append(class_prob)

    return model_eval


def compare_similarity_methods(method_max_concept_sim, visualization_output_dir, model0_name, model1_name, save=True,
                               show=False):
    out_folder = os.path.join(visualization_output_dir, 'method_comparison')
    os.makedirs(out_folder, exist_ok=True)

    method_dict = {'lasso_regression_c': 'CMCS (Pearson)', 'pearson': 'MCS (Pearson)'}

    pairs = list(combinations(method_max_concept_sim.keys(), 2))
    for pair in pairs:

        method1, method2 = pair

        method1_name = method1 + method_max_concept_sim[method1]['output_dir'].split(method1)[-1].replace('/', '_')
        method2_name = method2 + method_max_concept_sim[method2]['output_dir'].split(method2)[-1].replace('/', '_')

        method1_name = method_dict.get(method1, method1_name)
        method2_name = method_dict.get(method2, method2_name)

        rtoc1 = method_max_concept_sim[method1]['rtoc']
        rtoc2 = method_max_concept_sim[method2]['rtoc']
        ctor1 = method_max_concept_sim[method1]['ctor']
        ctor2 = method_max_concept_sim[method2]['ctor']

        rtoc1 = np.array(rtoc1)
        rtoc2 = np.array(rtoc2)
        ctor1 = np.array(ctor1)
        ctor2 = np.array(ctor2)

        fig, axes = plt.subplots(2, 1, squeeze=False)
        fig.set_size_inches(5, 10)
        ax = axes[0, 0]
        ax.scatter(rtoc1, rtoc2, alpha=0.5, color='black')
        ax.set_title(f'{model0_name} $\\rightarrow$ {model1_name}', fontsize=17)
        ax.set_xlabel(f'{method1_name}', fontsize=17)
        ax.set_ylabel(f'{method2_name}', fontsize=17)
        ax.set_xlim([-0.0, 1])
        ax.set_ylim([-0.0, 1])
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax = axes[1, 0]
        ax.scatter(ctor1, ctor2, alpha=0.5, color='black')
        ax.set_title(f'{model1_name} $\\rightarrow$ {model0_name}', fontsize=17)
        ax.set_xlabel(f'{method1_name}', fontsize=17)
        ax.set_ylabel(f'{method2_name}', fontsize=17)
        ax.set_xlim([-0.0, 1])
        ax.set_ylim([-0.0, 1])
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(out_folder, f'{method1_name}_{method2_name}.png'), dpi=400)
        if show:
            plt.show()


def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--importance_output_root', type=str, default='./')
    parser.add_argument('--eval_dataset', type=str, default='imagenet')
    parser.add_argument('--data_split', type=str, default='train')
    parser.add_argument('--visualize_summary_plot_comparison_indices', type=str, default='1')
    parser.add_argument('--visualize_baseline', action='store_true')
    parser.add_argument('--overlay_mean', action='store_true')
    parser.add_argument('--fontsize', type=int, default=16)
    parser.add_argument('--use_train_data', action='store_true')

    args = parser.parse_args()
    args.visualize_summary_plot_comparison_indices = [int(x) for x in
                                                      args.visualize_summary_plot_comparison_indices.split(',')]

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    dataset = config['dataset']
    num_images = config['num_images']
    num_image_repeats = config['num_image_repeats']
    transform_type = config['transform_type']
    seed = config['seed']
    set_seed(seed)
    comparison_methods = config['methods']
    comparison_name = config['comparison_name']
    only_last_layer = config['only_last_layer']
    patchify = config['patchify']
    args.comparison_save_name = comparison_name
    args.patchify = patchify

    comparison_output_dir = build_output_dir(args.comparison_output_root, 'concept_comparison', comparison_name)
    data_group_name, method_output_folders = process_config(config, comparison_output_dir)

    print(method_output_folders)

    out = build_model_comparison_param_dicts(args)
    param_dicts1 = out['param_dicts1']
    save_names1 = out['save_names1']
    param_dicts2 = out['param_dicts2']
    save_names2 = out['save_names2']
    model0_name = save_names1['model_name']
    model1_name = save_names2['model_name']

    model0_int_grad_out_dir, model1_int_grad_out_dir = build_importance_output_dir(args, config, save_names1, save_names2, data_group_name)
    igs = args.cross_model_image_group_strategy
    dataset_name = args.dataset_0

    fe_outs = []
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
        model = model_out['model']

        fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
        fe_outs.append(fe_out)

        # overwrite original num images (need original for accurately loading paths)
        param_dicts['num_images'] = args.cmigs_num_images

    image_group = create_image_group(strategy=igs, param_dicts=[param_dicts1, param_dicts2])

    # dataset = load_dataset(model, args.eval_dataset, split='val')
    # if args.eval_dataset == 'imagenet':
    #     labels = np.array([x[1] for x in dataset.active_samples])
    # else:
    #     raise NotImplementedError

    model_save_names = [save_names1['model_name'], save_names2['model_name']]
    # eval_dict = load_model_stats(args, dataset_name, image_group, param_dicts1, param_dicts2, model_save_names)
    eval_dict = None

    class_list = param_dicts1['class_list']
    # load integrated gradients ahead of time to calculate min and max accurately

    int_grad1_list = []
    int_grad0_list = []
    ig0_class_inds = []
    ig1_class_inds = []
    abs_max_0 = 0
    abs_max_1 = 0
    for class_idx in class_list:

        if os.path.exists(os.path.join(model0_int_grad_out_dir, f'{class_idx}.pth')):
            int_grad0 = torch.load(os.path.join(model0_int_grad_out_dir, f'{class_idx}.pth'))
            int_grad0 = -1 * int_grad0
            int_grad0_list.append(int_grad0)
            ig0_class_inds.append(class_idx)
            if int_grad0.mean(0).abs().max() > abs_max_0:
                abs_max_0 = int_grad0.mean(0).abs().max()

        if os.path.exists(os.path.join(model1_int_grad_out_dir, f'{class_idx}.pth')):
            int_grad1 = torch.load(os.path.join(model1_int_grad_out_dir, f'{class_idx}.pth'))
            int_grad1 = -1 * int_grad1
            int_grad1_list.append(int_grad1)
            ig1_class_inds.append(class_idx)
            if int_grad1.mean(0).abs().max() > abs_max_1:
                abs_max_1 = int_grad1.mean(0).abs().max()

    # with open(os.path.join(importance_output_dir, 'abs_max.json'), 'w') as f:
    #     json.dump({'0': abs_max_0.item(), '1': abs_max_1.item()}, f, indent=2)

    method_max_concept_sim = {}
    for mi, method_dict in enumerate(comparison_methods):
        method = method_dict['method']
        if method != 'lasso_regression_c' and method != 'lasso_regression':
            continue

        print("Method: ", method)
        method_output_folder = method_dict['method_output_folder']
        comparison_folder = method_output_folder.split('concept_comparison/')[-1]
        comparison_dir = method_output_folders[mi]
        visualization_output_dir = os.path.join(args.output_root, 'outputs', 'visualizations',
                                                'similarity_vs_importance' if not args.visualize_baseline else 'similarity_vs_importance_baseline',
                                                comparison_folder)
        # if os.path.exists(visualization_output_dir):
        #     print(f'Folder {visualization_output_dir} exists. Skipping...')
        #     continue
        viz_spc = mi in args.visualize_summary_plot_comparison_indices
        viz_spc = False
        os.makedirs(visualization_output_dir, exist_ok=True)
        plot_params = dict(show=False, save=True, visualization_output_dir=visualization_output_dir, method=method,
                           model0_name=model0_name, model1_name=model1_name,
                           model0_plot_name=ph.plot_names[model0_name], model1_plot_name=ph.plot_names[model1_name],
                           visualize_each_class=False, visualize_summary_plot_comparison=viz_spc,
                           concept_viz_dir1=save_names1['visualization_dir'],
                           concept_viz_dir2=save_names2['visualization_dir'],
                           eval_dict=eval_dict,
                           patchify=patchify,
                           use_train_data=args.use_train_data,
                           baseline=args.visualize_baseline,
                           overlay_mean=args.overlay_mean,
                           fontsize=args.fontsize,
                           )

        out = visualize_similarity_vs_importance(class_list, comparison_dir, fe_outs, eval_dict,
                                                 int_grad0_list, int_grad1_list, plot_params)
        rtoc_list = out['rtoc']
        ctor_list = out['ctor']
        method_max_concept_sim[method] = {'rtoc': rtoc_list, 'ctor': ctor_list, 'output_dir': visualization_output_dir}
        out.update(plot_params)
        out['class_list'] = class_list
        # visualize_similarity_samples(out['samples_class_lists'], eval_dict, out)
    comparison_folder = comparison_folder.rstrip('/')
    output_dir = os.path.join(args.output_root, 'outputs', 'visualizations', 'similarity_vs_importance',
                              comparison_folder.split(method)[0])
    compare_similarity_methods(method_max_concept_sim, output_dir, ph.plot_names[model0_name], ph.plot_names[model1_name], show=False)


if __name__ == '__main__':
    main()
