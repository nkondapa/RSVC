import matplotlib.pyplot as plt
import numpy as np
from src.utils import saving, model_loader, concept_extraction_helper as ceh
import json
import os
from tqdm import tqdm
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from compare_models import build_model_comparison_parser, set_seed, build_output_dir, process_config
from src.utils import plotting_helper as ph
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def compute_summary_similarity_matrix(methods, class_list, fe_outs, comparison_dirs):
    pbar = tqdm(class_list)
    mean_arr = np.zeros((len(class_list), len(fe_outs[0]['layer_names']), len(fe_outs[1]['layer_names'])))
    max_arr = np.zeros((len(class_list), len(fe_outs[0]['layer_names']), len(fe_outs[1]['layer_names'])))
    m1_mean_max = np.zeros((len(class_list), len(fe_outs[0]['layer_names']), len(fe_outs[1]['layer_names'])))
    m2_mean_max = np.zeros((len(class_list), len(fe_outs[0]['layer_names']), len(fe_outs[1]['layer_names'])))
    for ci, class_idx in enumerate(pbar):
        pbar.set_description(f'Class {class_idx}')

        for li, m0_layer in enumerate(fe_outs[0]['layer_names']):  # reverse order to start from the last layer
            # print(Ui.shape)
            for lj, m1_layer in enumerate(
                    fe_outs[1]['layer_names']):  # reverse order to start from the last layer

                # Compare coefficient matrices
                for mi, method in enumerate(methods):
                    p = os.path.join(comparison_dirs[mi], f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl')
                    if not os.path.isfile(p):
                        print(f'Skipping {p}')
                        continue
                    with open(p, 'rb') as f:
                        statistic_dict = pkl.load(f)
                    if statistic_dict is None:
                        continue
                    _arr = convert_to_array(statistic_dict)
                    mean_arr[ci, li, lj] = np.nanmean(_arr)
                    max_arr[ci, li, lj] = np.nanmax(_arr)
                    m1_mean_max[ci, li, lj] = np.nanmean(np.nanmax(_arr, axis=1))
                    m2_mean_max[ci, li, lj] = np.nanmean(np.nanmax(_arr, axis=0))

    return dict(mean_matrix=mean_arr, max_matrix=max_arr, m1_mean_max_matrix=m1_mean_max, m2_mean_max_matrix=m2_mean_max)


def visualize_summary_similarity_matrix_clusters(summary_matrices, metadata, fe_out, output_dir, show=True, save=False):
    max_matrix = summary_matrices['max_matrix']
    c, l1, l2 = max_matrix.shape

    num_clusters = 6
    km = KMeans(n_clusters=6)
    km.fit(max_matrix.reshape(c, -1))

    red = PCA(n_components=2)
    max_sim2d = red.fit_transform(max_matrix.reshape(c, -1))
    fig, ax = plt.subplots(1, 1)
    for i in range(num_clusters):
        ax.scatter(max_sim2d[km.labels_ == i, 0], max_sim2d[km.labels_ == i, 1], label=f'Cluster {i}')
    # ax.scatter(max_sim2d[:, 0], max_sim2d[:, 1])
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_title('Concept Similarity Decomp')
    plt.legend()
    plt.show()

    red = TSNE(n_components=2)
    max_sim2d = red.fit_transform(max_matrix.reshape(c, -1))
    fig, ax = plt.subplots(1, 1)
    for i in range(num_clusters):
        ax.scatter(max_sim2d[km.labels_ == i, 0], max_sim2d[km.labels_ == i, 1], label=f'Cluster {i}')
    # ax.scatter(max_sim2d[:, 0], max_sim2d[:, 1])
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_title('Concept Similarity Decomp')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(num_clusters, 1)
    fig.set_size_inches(9, 3 * num_clusters)
    for i in range(num_clusters):
        ax = axes[i]
        ax.imshow(max_matrix[km.labels_ == i].mean(0))
        ax.set_title(f'Cluster {i} : N={np.sum(km.labels_ == i)}')
    plt.show()
    plt.close()


def visualize_summary_similarity_matrices(summary_matrices, metadata, fe_out, output_dir, plot_params, show=True, save=False):


    model0_plot_name = plot_params['model0_plot_name']
    model1_plot_name = plot_params['model1_plot_name']
    transpose = plot_params['transpose_plot']
    ext = plot_params.get('ext', 'png')
    dpi = 300 if ext == 'png' else None

    if transpose:
        for k, v in summary_matrices.items():
            summary_matrices[k] = np.transpose(v, (0, 2, 1))
        tmp = model0_plot_name
        model0_plot_name = model1_plot_name
        model1_plot_name = tmp

    dim1, dim2 = summary_matrices['mean_matrix'].shape[1:]
    wide_plot = dim1 <= dim2
    if wide_plot:
        fig_size = (6, 3)
    else:
        fig_size = (3, 6)
    print(dim1, dim2, wide_plot, fig_size)

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    mean_arr_max_similarity = {}
    for i, (arr_name, arr) in enumerate(summary_matrices.items()):
        _mean_arr = np.nanmean(arr, axis=0)
        # print(_mean_arr.max(0))
        mean_arr_max_similarity[arr_name] = [_mean_arr.max(1), _mean_arr.max(0)]
        ax = axes.flatten()[i]
        im = ax.imshow(_mean_arr, cmap='viridis', origin='lower')
        plt.colorbar(im)
    if save:
        plt.savefig(os.path.join(output_dir, f'all_matrix_types.{ext}'), dpi=dpi)
    if show:
        plt.show()

    fig, axes = plt.subplots(2, 2)
    for i, (arr_name, arr) in enumerate(mean_arr_max_similarity.items()):
        ax = axes.flatten()[i]
        ax.plot(arr[0])
    if save:
        plt.savefig(os.path.join(output_dir, f'mean_arr_max_similarity_0.{ext}'), dpi=dpi)
    if show:
        plt.show()

    fig, axes = plt.subplots(2, 2)
    for i, (arr_name, arr) in enumerate(mean_arr_max_similarity.items()):
        ax = axes.flatten()[i]
        ax.plot(arr[1])
    if save:
        plt.savefig(os.path.join(output_dir, f'mean_arr_max_similarity_1.{ext}'), dpi=dpi)
    if show:
        plt.show()


    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(*fig_size)
    ax = axes
    im = ax.imshow(np.nanmean(summary_matrices['max_matrix'], axis=0), origin='lower')

    ax.set_xlabel(f'{model1_plot_name} Layers', fontsize=14)
    ax.set_ylabel(f'{model0_plot_name} Layers', fontsize=14)
    ax.set_title('Maximum Concept Similarity', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # plt.colorbar(im)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'max_similarity_matrix.{ext}'), dpi=dpi)
    if show:
        plt.show()

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(*fig_size)
    ax = axes
    im = ax.imshow(np.nanmean(summary_matrices['m1_mean_max_matrix'], axis=0), origin='lower')
    ax.set_xlabel(f'{model1_plot_name} Layers', fontsize=14)
    ax.set_ylabel(f'{model0_plot_name} Layers', fontsize=14)
    ax.set_title(f'{model0_plot_name} MMCS', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'm1_mean_max_similarity_matrix.{ext}'), dpi=dpi)
    if show:
        plt.show()

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(*fig_size)
    ax = axes
    im = ax.imshow(np.nanmean(summary_matrices['m2_mean_max_matrix'], axis=0), origin='lower')
    ax.set_xlabel(f'{model1_plot_name} Layers', fontsize=14)
    ax.set_ylabel(f'{model0_plot_name} Layers', fontsize=14)
    ax.set_title(f'{model1_plot_name} MMCS', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'm2_mean_max_similarity_matrix.{ext}'), dpi=dpi)
    if show:
        plt.show()

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(*fig_size)
    ax = axes
    mmcs = np.nanmean(summary_matrices['m1_mean_max_matrix'], axis=0) + np.nanmean(summary_matrices['m2_mean_max_matrix'], axis=0)
    mmcs = mmcs / 2
    im = ax.imshow(mmcs, origin='lower')
    ax.set_xlabel(f'{model1_plot_name}', fontsize=14)
    ax.set_ylabel(f'{model0_plot_name}', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'mean_max_similarity_matrix.{ext}'), dpi=dpi)
    if show:
        plt.show()

def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--transpose_plot', action='store_true', help='Reverse the direction of the plot')
    args = parser.parse_args()

    out = build_model_comparison_param_dicts(args)
    param_dicts1 = out['param_dicts1']
    save_names1 = out['save_names1']
    param_dicts2 = out['param_dicts2']
    save_names2 = out['save_names2']

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    seed = config['seed']
    set_seed(seed)
    comparison_methods = config['methods']
    comparison_name = config['comparison_name']
    args.comparison_save_name = comparison_name

    comparison_output_dir = build_output_dir(args.comparison_output_root, 'concept_comparison', comparison_name)
    data_group_name, method_output_folders = process_config(config, comparison_output_dir)

    fe_outs = []
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
        model = model_out['model']

        fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
        fe_outs.append(fe_out)

        # overwrite original num images (need original for accurately loading paths)
        param_dicts['num_images'] = args.cmigs_num_images

    plot_params = {'model0_plot_name': ph.plot_names[save_names1['model_name']],
                   'model1_plot_name': ph.plot_names[save_names2['model_name']],
                   'transpose_plot': args.transpose_plot,
                   'ext': 'png'
                   }

    for mi, method_dict in enumerate(comparison_methods):
        method = method_dict['method']
        comparison_dir = method_output_folders[mi]
        force_compute = True
        output_dir = os.path.join(args.output_root, 'outputs', 'data', 'concept_comparison', args.comparison_save_name, data_group_name)
        if not os.path.isfile(os.path.join(output_dir, 'summary_matrices', f'{method}_summary_matrices.npy')) or force_compute:
            os.makedirs(os.path.join(output_dir, 'summary_matrices'), exist_ok=True)
            summary_matrices = compute_summary_similarity_matrix([method], param_dicts1['class_list'], fe_outs, [comparison_dir])
            with open(os.path.join(output_dir, 'summary_matrices', f'{method}_summary_matrices.npy'), 'wb') as f:
                np.save(f, summary_matrices, allow_pickle=True)
        else:
            with open(os.path.join(output_dir, 'summary_matrices', f'{method}_summary_matrices.npy'), 'rb') as f:
                summary_matrices = np.load(f, allow_pickle=True).tolist()

        visualizations_dir = os.path.join(args.output_root, 'outputs', 'visualizations', 'layerwise_concept_comparisons', args.comparison_save_name, data_group_name, method)
        os.makedirs(visualizations_dir, exist_ok=True)
        visualize_summary_similarity_matrices(summary_matrices, out, fe_outs, visualizations_dir, plot_params, show=False, save=True)


if __name__ == '__main__':
    main()
