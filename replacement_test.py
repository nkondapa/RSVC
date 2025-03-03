import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from src.utils import saving, model_loader, concept_extraction_helper as ceh

import json
import os
from tqdm import tqdm

from extract_model_activations import create_image_group
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from concept_integrated_gradients import build_output_dir as build_importance_output_dir
# from concept_replacement_test import load_dataset
from compare_models import build_model_comparison_parser, set_seed, build_output_dir, process_config
from itertools import combinations

from src.utils import plotting_helper as ph
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.mixture import GaussianMixture
import sklearn


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


def process_comparison_dict(method, comparison_dir, class_idx, m0_layer, m1_layer, patchify=False):
    '''

    :param method:
    :param comparison_dict:
    :return: max_concept_match_rtoc, max_concept_match_ctor
    max_concept_match_rtoc:
        correlation: given a concept in model1 what is the best match from model2  (max in each column)
        regression: how well does model2 predict a model1 concept
    max_concept_match_ctor:
        correlation: given a concept in model2 what is the best match from model1  (max in each column)
        regression: how well does model1 predict a model2 concept
    '''

    def load_comp_dict():
        p = os.path.join(comparison_dir, f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl')
        with open(p, 'rb') as f:
            comparison_dict = pkl.load(f)
        return comparison_dict

    def load_regression_eval_dict(patched=False):
        folder_name = 'regression_evaluation' if not patched else 'regression_evaluation_patched'
        eval_dir = comparison_dir.replace('concept_comparison', folder_name)
        p = os.path.join(eval_dir, f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl')
        try:
            with open(p, 'rb') as f:
                eval_dict = pkl.load(f)
        except FileNotFoundError:
            eval_dict = None
        return eval_dict

    if method in ['lasso_regression', 'lasso_regression_c', 'k_nearest_neighbors', 'concept_lasso_regression',
                    'concept_lasso_regression_c']:
        eval_dict = load_regression_eval_dict(patchify)
        return eval_dict

    else:
        print()
        raise ValueError(f'Unknown method {method}')



def visualize_replacement_test(class_list, comparison_dir, fe_outs, eval_dict,
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
    class_specific_viz_folder_model0 = None
    class_specific_viz_folder_model1 = None
    summary_plot_comparison_dir = None
    patchify = plot_params.get('patchify', False)
    abs_max0 = plot_params['ig0_max']
    abs_max1 = plot_params['ig1_max']
    pbar = tqdm(class_list)
    m0_layer = fe_outs[0]['layer_names'][-1]
    m1_layer = fe_outs[1]['layer_names'][-1]
    num_concepts = int_grad0[0].shape[1]

    ci = 0
    stats0 = {'pearsonr': [], 'spearmanr': [], 'l2': [], 'kl': [], 'match_acc': [], 'ig': []}
    stats1 = {'pearsonr': [], 'spearmanr': [], 'l2': [], 'kl': [], 'match_acc': [], 'ig': []}
    stats00 = {'pearsonr': [], 'spearmanr': [], 'l2': [], 'kl': [], 'match_acc': [], 'ig': []}
    stats11 = {'pearsonr': [], 'spearmanr': [], 'l2': [], 'kl': [], 'match_acc': [], 'ig': []}
    valid_class_indices = []
    for class_idx in pbar:

        out = process_comparison_dict(method, comparison_dir, class_idx, m0_layer, m1_layer, patchify=patchify)
        if out is None:
            continue
        valid_class_indices.append(class_idx)
        scores = out['scores']

        ig0 = int_grad0[ci] / abs_max0
        ig1 = int_grad1[ci] / abs_max1
        print(f'{ci} / {len(int_grad0)}')
        print(f'{ci} / {len(int_grad1)}')

        acc_m0_recon = out['analysis_data']['acc_m0_recon']
        acc_m1_recon = out['analysis_data']['acc_m1_recon']

        score_name = ['2to1', '1to2', '1to1', '2to2']
        ig_opt = [ig0, ig1, ig0, ig1]
        for si, _stat_dict in enumerate([stats0, stats1, stats00, stats11]):
            sn = score_name[si]
            _stat_dict['pearsonr'].extend(list(scores['pearsonr'][sn]))
            _stat_dict['spearmanr'].extend(list(scores['spearmanr'][sn]))
            _stat_dict['l2'].extend(list(scores['replacement_l2'][sn]))
            _stat_dict['kl'].extend(list(scores['replacement_kl'][sn]))
            _stat_dict['match_acc'].extend(list(scores['replacement_match_acc'][sn]))
            _stat_dict['ig'].extend(list(ig_opt[si].mean(0)))

        ci += 1

    ind_to_class_concept = {}
    c = 0
    for class_idx in valid_class_indices:
        for i in range(num_concepts):
            ind_to_class_concept[c] = (class_idx, i)
            c += 1

    for stats in [stats0, stats1, stats00, stats11]:
        for key in stats:
            stats[key] = np.array(stats[key])

    stat_names = ['2to1', '1to2']
    regression_out = {}

    samples = {"1to2": {}, "2to1": {}}
    samples_data = {"1to2": {}, "2to1": {}}
    for si, (stats, stats_base) in enumerate([(stats0, stats00), (stats1, stats11)]):
        delta_p = stats['pearsonr'] - stats_base['pearsonr']
        # delta_p = stats_base['pearsonr'] - stats['pearsonr']
        delta_l2 = stats['l2'] - stats_base['l2']
        delta_kl = stats['kl'] - stats_base['kl']
        delta_ma = stats['match_acc'] - stats_base['match_acc']
        samp_key = stat_names[si]
        sorted_kl = np.argsort(delta_kl)[::-1]
        kl_thresh = delta_kl[sorted_kl[int(0.25 * len(sorted_kl))]]
        for k, (_c, _con) in [(k, ind_to_class_concept[k]) for k in delta_kl.argsort()[::-1][:15]]:
            if _c not in samples[samp_key]:
                samples[samp_key][_c] = []
            samples_data[samp_key][str((_c, _con))] = {'kl': delta_kl[k], 'l2': delta_l2[k], 'ma': delta_ma[k],
                                                       'dp': delta_p[k]}
            samples[samp_key][_c].append(_con)

        inds = np.arange(delta_p.shape[0])[delta_kl > kl_thresh]
        low_sim_above_thresh_inds = inds[delta_p[delta_kl > kl_thresh].argsort()]

        plt.figure()
        plt.scatter(delta_p[inds], delta_kl[inds], alpha=0.5)
        plt.title(f'{stat_names[si]}')
        if save:
            plt.savefig(os.path.join(visualization_output_dir, f'{stat_names[si]}_thresholded_delta_scatter.png'), dpi=600)
        if show:
            plt.show()

        for k, (_c, _con) in [(k, ind_to_class_concept[k]) for k in low_sim_above_thresh_inds[:15]]:
            if _c not in samples[samp_key]:
                samples[samp_key][_c] = []
            samples_data[samp_key][str((_c, _con))] = {'kl': delta_kl[k], 'l2': delta_l2[k], 'ma': delta_ma[k],
                                                       'dp': delta_p[k]}
            samples[samp_key][_c].append(_con)

        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(9, 3)
        for ax in axes.flatten():
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[0].scatter(delta_p, delta_l2, alpha=0.5, cmap='magma', c=np.abs(stats['ig']), vmin=0, vmax=1)
        axes[1].scatter(delta_p, delta_kl, alpha=0.5, cmap='magma', c=np.abs(stats['ig']), vmin=0, vmax=1)
        im = axes[2].scatter(delta_p, delta_ma, alpha=0.5, cmap='magma', c=np.abs(stats['ig']), vmin=0, vmax=1)
        # add colorbar to last axis for all subplots
        for ax in axes.flatten():
            ax.set_xlabel(r'$\Delta$ Pearson', fontsize=14)
        axes[0].set_ylabel(r'$\Delta$ L2', fontsize=14)
        axes[1].set_ylabel(r'$\Delta$ KL', fontsize=14)
        axes[2].set_ylabel(r'$\Delta$ MatchAcc', fontsize=14)
        if stat_names[si] == '2to1':
            plt.suptitle(f'{model0_plot_name}' + r'$\rightarrow$' + f'{model1_plot_name}', fontsize=14)
        else:
            plt.suptitle(f'{model1_plot_name}' + r'$\rightarrow$' + f'{model0_plot_name}', fontsize=14)
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.solids.set(alpha=1)
        if save:
            plt.savefig(os.path.join(visualization_output_dir, f'{stat_names[si]}_pearson_delta_to_metric.png'), dpi=600)
            plt.savefig(os.path.join(visualization_output_dir, f'{stat_names[si]}_pearson_delta_to_metric.pdf'))
        if show:
            plt.show()

        inds = np.arange(len(stats1['ig']))
        method = 'hist'
        if method == 'linear':
            lr = sklearn.linear_model.LinearRegression()
        else:
            lr = HistGradientBoostingRegressor(min_samples_leaf=500)
        train_inds = np.random.choice(inds, int(0.8 * len(inds)), replace=False)
        test_inds = np.setdiff1d(inds, train_inds)
        X_pearson = np.stack([delta_p, np.abs(stats['ig']), delta_p * np.abs(stats['ig'])], axis=1)
        Y = np.stack([delta_l2, delta_kl, delta_ma], axis=1)
        if not method == 'linear':
            Y = Y[:, [1]]
        X_train = X_pearson[train_inds]
        y_train = Y[train_inds]
        X_test = X_pearson[test_inds]
        y_test = Y[test_inds]
        data = {}
        var_names = ['Similarity', 'Importance', 'Similarity * Importance']
        target_names = ['L2', 'KL', 'MatchAcc']
        for var_inds in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]:
            data[str(var_inds)] = {}
            print(var_inds)
            if method == 'linear':
                lr.fit(X_train[:, var_inds], y_train)
                y_pred = lr.predict(X_test[:, var_inds])
                for i in range(Y.shape[1]):
                    data[str(var_inds)][target_names[i]] = sklearn.metrics.r2_score(y_test[:, i], y_pred[:, i])
            else:
                lr.fit(X_train[:, var_inds], y_train.flatten())
                score = lr.score(X_test[:, var_inds], y_test.flatten())
                print(score)
                data[str(var_inds)]['KL'] = score
                # print(lr.feature_importances_)
            print()
        print()
        regression_out[stat_names[si]] = data

    sn = comparison_dir.split('concept_comparison/')[-1].split('/')[0]
    os.makedirs(os.path.join("replacement_test_selected_samples"), exist_ok=True)
    with open(os.path.join("replacement_test_selected_samples", sn + '.json'), 'w') as f:
        json.dump(samples, f, indent=2)

    with open(os.path.join(visualization_output_dir, 'selected_samples_stats.json'), 'w') as f:
        json.dump(samples_data, f, indent=2)

    with open(os.path.join(visualization_output_dir, 'regression_out.json'), 'w') as f:
        json.dump(regression_out, f, indent=2)

    stat_dict_names = ['2to1', '1to2', '1to1', '2to2']
    cmap = plt.get_cmap('magma')
    alpha = 1
    # for si, stats in enumerate([stats0, stats1, stats00, stats11]):
    for si, (stats, stats_base) in enumerate([(stats0, stats00), (stats1, stats11)]):
        fig, axes = plt.subplots(2, 3)
        fig.set_size_inches(12, 8)
        inds = np.arange(len(stats['ig']))
        # mask = np.abs(stats['ig']) > 0.4
        # inds = inds[mask]
        # inds = np.argsort(np.abs(stats['ig']))

        ax = axes[0, 0]
        ax.scatter(stats['pearsonr'][inds], stats['l2'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.scatter(stats_base['pearsonr'][inds], stats_base['l2'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Pearson')
        ax.set_ylabel('L2')
        ax = axes[0, 1]
        ax.scatter(stats['pearsonr'][inds], stats['kl'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.scatter(stats_base['pearsonr'][inds], stats_base['kl'][inds], cmap=cmap, c=np.abs(stats_base['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Pearson')
        ax.set_ylabel('KL')
        ax = axes[0, 2]
        ax.scatter(stats['pearsonr'][inds], stats['match_acc'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.scatter(stats_base['pearsonr'][inds], stats_base['match_acc'][inds], cmap=cmap, c=np.abs(stats_base['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Pearson')
        ax.set_ylabel('MatchAcc')

        ax = axes[1, 0]
        ax.scatter(stats['spearmanr'][inds], stats['l2'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Spearman')
        ax.set_ylabel('L2')
        ax = axes[1, 1]
        ax.scatter(stats['spearmanr'][inds], stats['kl'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Spearman')
        ax.set_ylabel('KL')
        ax = axes[1, 2]
        ax.scatter(stats['spearmanr'][inds], stats['match_acc'][inds], cmap=cmap, c=np.abs(stats['ig'][inds]), vmin=0, vmax=1, alpha=alpha)
        ax.set_xlabel('Spearman')
        ax.set_ylabel('MatchAcc')
        for ax in axes.flatten():
            ax.set_xlim(0.5, 1)
        plt.suptitle(f'{stat_dict_names[si]}')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(visualization_output_dir, f'{stat_dict_names[si]}_stats.png'))
        if show:
            plt.show()

        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        axes[0].scatter(np.abs(stats['ig']), stats['l2'], alpha=alpha, vmin=stats['pearsonr'].min(), cmap='magma', c=stats['pearsonr'])
        axes[1].scatter(np.abs(stats['ig']), stats['kl'], alpha=alpha, vmin=stats['pearsonr'].min(), cmap='magma', c=stats['pearsonr'])
        axes[2].scatter(np.abs(stats['ig']), stats['match_acc'], alpha=alpha, vmin=stats['pearsonr'].min(), cmap='magma', c=stats['pearsonr'])
        if save:
            plt.savefig(os.path.join(visualization_output_dir, f'{stat_dict_names[si]}_imp_vs_metric.png'))
        if show:
            plt.show()

        print(stat_dict_names[si])
        method = 'gbm'
        if method == 'linear':
            lr = sklearn.linear_model.LinearRegression()
        else:
            lr = HistGradientBoostingRegressor(min_samples_leaf=500)
        train_inds = np.random.choice(inds, int(0.8 * len(inds)), replace=False)
        test_inds = np.setdiff1d(inds, train_inds)
        all_sim = np.concatenate([stats['pearsonr'], stats_base['pearsonr']])
        all_l2 = np.concatenate([stats['l2'], stats_base['l2']])
        all_kl = np.concatenate([stats['kl'], stats_base['kl']])
        all_ma = np.concatenate([stats['match_acc'], stats_base['match_acc']])
        X_pearson = np.stack([stats['pearsonr'], np.abs(stats['ig']), stats['pearsonr'] * np.abs(stats['ig'])], axis=1)
        X_spearman = np.stack([stats['spearmanr'], np.abs(stats['ig']), stats['spearmanr'] * np.abs(stats['ig'])], axis=1)
        Y = np.stack([stats['l2'], stats['kl'], stats['match_acc']], axis=1)
        if not method == 'linear':
            Y = Y[:, [1]]
        X_train = X_pearson[train_inds]
        y_train = Y[train_inds]
        X_test = X_pearson[test_inds]
        y_test = Y[test_inds]
        data = {}
        for var_inds in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]:
            data[str(var_inds)] = []
            print(var_inds)
            if method == 'linear':
                lr.fit(X_train[:, var_inds], y_train)
                y_pred = lr.predict(X_test[:, var_inds])
                for i in range(Y.shape[1]):
                    print(sklearn.metrics.r2_score(y_test[:, i], y_pred[:, i]))
                print(lr.coef_)
            else:
                lr.fit(X_train[:, var_inds], y_train.flatten())
                score = lr.score(X_test[:, var_inds], y_test.flatten())
                print(score)
                # print(lr.feature_importances_)
            print()
        print()

    return None

def load_model_stats(args, dataset_name, image_group, param_dicts1, param_dicts2, model_save_names):
    model_eval = {}
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_save_name = model_save_names[mi]
        # path = f"model_evaluation/{dataset_name}/{model_name}_probs_{args.data_split}.pth"
        # print(path)
        # probs = torch.load(path)
        with open(f'model_evaluation/{dataset_name}/{model_name}_{args.data_split}.json', 'r') as f:
            eval_dict = json.load(f)
        with open(f'model_evaluation/{dataset_name}/{model_name}_stats_{args.data_split}.json', 'r') as f:
            stats = json.load(f)

        print(f"Model: {model_save_name}")
        img_paths = np.array(list(eval_dict['predictions'].keys()))
        labels = np.array(eval_dict['labels'])
        model_eval[model_save_name] = {'mean': [], 'sample': [], 'stats': stats}
        # for class_idx in np.unique(labels):
        #     # mask = labels == class_idx
        #     # mask = np.array([path in sampled_paths for path in img_paths])
        #     # class_paths = img_paths[mask]
        #     class_paths = image_group[class_idx]
        #     class_prob = np.array([probs[path][class_idx] for path in class_paths])
        #     model_eval[model_save_name]['mean'].append(class_prob.mean())
        #     model_eval[model_save_name]['sample'].append(class_prob)

    return model_eval


def compare_similarity_methods(method_max_concept_sim, visualization_output_dir, model0_name, model1_name, save=True,
                               show=False):
    out_folder = os.path.join(visualization_output_dir, 'method_comparison')
    os.makedirs(out_folder, exist_ok=True)

    pairs = list(combinations(method_max_concept_sim.keys(), 2))
    for pair in pairs:

        method1, method2 = pair

        method1_name = method1 + method_max_concept_sim[method1]['output_dir'].split(method1)[-1].replace('/', '_')
        method2_name = method2 + method_max_concept_sim[method2]['output_dir'].split(method2)[-1].replace('/', '_')

        rtoc1 = method_max_concept_sim[method1]['rtoc']
        rtoc2 = method_max_concept_sim[method2]['rtoc']
        ctor1 = method_max_concept_sim[method1]['ctor']
        ctor2 = method_max_concept_sim[method2]['ctor']

        rtoc1 = np.array(rtoc1)
        rtoc2 = np.array(rtoc2)
        ctor1 = np.array(ctor1)
        ctor2 = np.array(ctor2)

        fig, axes = plt.subplots(2, 1, squeeze=False)
        fig.set_size_inches(10, 8)
        ax = axes[0, 0]
        ax.scatter(rtoc1, rtoc2, alpha=0.5, color='blue')
        ax.set_title(f'{model0_name} -> {model1_name} ')
        ax.set_xlabel(f'{method1_name}')
        # ax.set_ylabel(f'{method2_name}')
        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax = axes[1, 0]
        ax.scatter(ctor1, ctor2, alpha=0.5, color='blue')
        ax.set_title(f'{model1_name} -> {model0_name} ')
        ax.set_xlabel(f'{method1_name}')
        ax.set_ylabel(f'{method2_name}')
        ax.set_xlim([-0.2, 1])
        ax.set_ylim([-0.2, 1])
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')

        if save:
            plt.savefig(os.path.join(out_folder, f'{method1_name}_{method2_name}.png'))
        if show:
            plt.show()


def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--importance_output_root', type=str, default='./')
    parser.add_argument('--eval_dataset', type=str, default='imagenet')
    parser.add_argument('--data_split', type=str, default='train')
    parser.add_argument('--visualize_summary_plot_comparison_indices', type=str, default='1')

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

    model_save_names = [save_names1['model_name'], save_names2['model_name']]

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

    method_max_concept_sim = {}
    for mi, method_dict in enumerate(comparison_methods):
        method = method_dict['method']
        if method != 'lasso_regression' and method != 'lasso_regression_c':
            # replacement test only works with regression based comparisons
            continue

        print("Method: ", method)
        method_output_folder = method_dict['method_output_folder']
        comparison_folder = method_output_folder.split('concept_comparison/')[-1]
        comparison_dir = method_output_folders[mi]
        visualization_output_dir = os.path.join(args.output_root, 'outputs', 'visualizations',
                                                'replacement_test',
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
                           ig0_max=abs_max_0, ig1_max=abs_max_1,
                           patchify=patchify)

        out = visualize_replacement_test(class_list, comparison_dir, fe_outs, eval_dict,
                                         int_grad0_list, int_grad1_list, plot_params)


if __name__ == '__main__':
    main()
