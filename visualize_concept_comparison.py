import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHook
import json
import os
from tqdm import tqdm

import copy
from extract_model_activations import create_image_group, _batch_inference
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from src.utils.funcs import _batch_inference, correlation_comparison, load_concepts, compute_concept_coefficients
from scipy.stats import pearsonr, spearmanr, rankdata
from compare_models import build_model_comparison_parser, set_seed, build_output_dir, process_config
import torchvision
from PIL import Image, ImageOps
from compare_models import standardize, unstandardize
from data.imagenet.imagenet_classes import imagenet_classes
from data.nabirds.classname_to_label import nabirds_classes
from src.utils.model_loader import split_model
import torch.nn.functional as F
from src import eval_model


from mpl_toolkits.axes_grid1 import ImageGrid
'''
Steps
1. Load activations for a class, model, layer
    - Re-create image patches
2. Load learned regression model for compared model
3. Regression on activations to predict concept coefficients
4. Save top-k image patches for real and pred
5. Save concept coefficients for real and pred

'''


def make_axes_invisible(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _select_class_and_load_images(image_path_list, data_root, crop_and_resize_fn, transform, repeat=1):
    sel_paths = image_path_list
    gt_labels = np.array([path.split('/')[-2] for path in sel_paths])

    images_preprocessed = []
    cropped_images = []
    crop_sizes = []
    for i, img_path in enumerate(sel_paths):
        img = Image.open(os.path.join(data_root, img_path.lstrip('/'))).convert('RGB')
        # print(img.size)
        # img = torchvision.transforms.Resize((224, 224))(img)
        crop_img, resized_img = crop_and_resize_fn(img)
        # print(crop_img.size)
        img = transform(resized_img)
        images_preprocessed.append(img)
        crop_sizes.append(crop_img.size)
        cropped_images.append(torchvision.transforms.ToTensor()(torchvision.transforms.Resize((64, 64))(crop_img)))
    cropped_images = torch.stack(cropped_images, 0)
    images_preprocessed = torch.stack(images_preprocessed, 0)

    out = {
        'image_paths': sel_paths,
        'gt_labels': gt_labels,
        'images_preprocessed': images_preprocessed,
        'num_images': len(sel_paths),
        'image_size': images_preprocessed.shape[2],
        'crop_sizes': crop_sizes,
        'cropped_images': cropped_images,
    }
    return out


def process_comparison_dict(comparison_dict, mean_lr=False):
    if 'lr1to2' in comparison_dict:
        return comparison_dict
    elif mean_lr:
        nd = {
            'lr1to1_pearson': [],
            'lr2to2_pearson': [],
            'lr1to2_pearson': [],
            'lr2to1_pearson': [],
            'lr1to1_coef': [],
            'lr2to2_coef': [],
            'lr1to2_coef': [],
            'lr2to1_coef': [],
            'lr1to1_intercept': [],
            'lr2to2_intercept': [],
            'lr1to2_intercept': [],
            'lr2to1_intercept': [],
            'act1_mean': [],
            'act1_std': [],
            'act2_mean': [],
            'act2_std': [],
            'U1_mean': [],
            'U1_std': [],
            'U2_mean': [],
            'U2_std': [],
        }
        for k in comparison_dict:
            cd = comparison_dict[k]
            nd['lr1to1_pearson'].append(cd['lr1to1_pearson'])
            nd['lr2to2_pearson'].append(cd['lr2to2_pearson'])
            nd['lr1to2_pearson'].append(cd['lr1to2_pearson'])
            nd['lr2to1_pearson'].append(cd['lr2to1_pearson'])
            nd['lr1to1_coef'].append(cd['lr1to1'].coef_)
            nd['lr2to2_coef'].append(cd['lr2to2'].coef_)
            nd['lr1to2_coef'].append(cd['lr1to2'].coef_)
            nd['lr2to1_coef'].append(cd['lr2to1'].coef_)
            nd['lr1to1_intercept'].append(cd['lr1to1'].intercept_)
            nd['lr2to2_intercept'].append(cd['lr2to2'].intercept_)
            nd['lr1to2_intercept'].append(cd['lr1to2'].intercept_)
            nd['lr2to1_intercept'].append(cd['lr2to1'].intercept_)
            nd['act1_mean'].append(cd.get('act1_mean', [0]))
            nd['act1_std'].append(cd.get('act1_std', [0]))
            nd['act2_mean'].append(cd.get('act2_mean', [0]))
            nd['act2_std'].append(cd.get('act2_std', [0]))
            nd['U1_mean'].append(cd.get('U1_mean', [0]))
            nd['U1_std'].append(cd.get('U1_std', [0]))
            nd['U2_mean'].append(cd.get('U2_mean', [0]))
            nd['U2_std'].append(cd.get('U2_std', [0]))

        # compute mean
        for k in nd:
            nd[k] = np.mean(np.stack(nd[k]), axis=0)

        nd['lr1to1'] = copy.copy(cd['lr1to1'])
        nd['lr2to2'] = copy.copy(cd['lr2to2'])
        nd['lr1to2'] = copy.copy(cd['lr1to2'])
        nd['lr2to1'] = copy.copy(cd['lr2to1'])

        nd['lr1to1'].coef_ = nd['lr1to1_coef']
        nd['lr2to2'].coef_ = nd['lr2to2_coef']
        nd['lr1to2'].coef_ = nd['lr1to2_coef']
        nd['lr2to1'].coef_ = nd['lr2to1_coef']

        nd['lr1to1'].intercept_ = nd['lr1to1_intercept']
        nd['lr2to2'].intercept_ = nd['lr2to2_intercept']
        nd['lr1to2'].intercept_ = nd['lr1to2_intercept']
        nd['lr2to1'].intercept_ = nd['lr2to1_intercept']

        return nd
    else:
        nd = {
            'lr1to1_pearson': [],
            'lr2to2_pearson': [],
            'lr1to2_pearson': [],
            'lr2to1_pearson': [],
            'lr1to1': [],
            'lr2to2': [],
            'lr1to2': [],
            'lr2to1': [],
            'lr1to1_coef': [],
            'lr2to2_coef': [],
            'lr1to2_coef': [],
            'lr2to1_coef': [],
            'lr1to1_intercept': [],
            'lr2to2_intercept': [],
            'lr1to2_intercept': [],
            'lr2to1_intercept': [],
            'act1_mean': [],
            'act1_std': [],
            'act2_mean': [],
            'act2_std': [],
            'U1_mean': [],
            'U1_std': [],
            'U2_mean': [],
            'U2_std': [],
        }
        for k in comparison_dict:
            cd = comparison_dict[k]
            nd['lr1to1_pearson'].append(cd['lr1to1_pearson'])
            nd['lr2to2_pearson'].append(cd['lr2to2_pearson'])
            nd['lr1to2_pearson'].append(cd['lr1to2_pearson'])
            nd['lr2to1_pearson'].append(cd['lr2to1_pearson'])
            nd['lr1to1_coef'].append(cd['lr1to1'].coef_)
            nd['lr2to2_coef'].append(cd['lr2to2'].coef_)
            nd['lr1to2_coef'].append(cd['lr1to2'].coef_)
            nd['lr2to1_coef'].append(cd['lr2to1'].coef_)
            nd['lr1to1_intercept'].append(cd['lr1to1'].intercept_)
            nd['lr2to2_intercept'].append(cd['lr2to2'].intercept_)
            nd['lr1to2_intercept'].append(cd['lr1to2'].intercept_)
            nd['lr2to1_intercept'].append(cd['lr2to1'].intercept_)
            nd['act1_mean'].append(cd.get('act1_mean', [0]))
            nd['act1_std'].append(cd.get('act1_std', [0]))
            nd['act2_mean'].append(cd.get('act2_mean', [0]))
            nd['act2_std'].append(cd.get('act2_std', [0]))
            nd['U1_mean'].append(cd.get('U1_mean', [0]))
            nd['U1_std'].append(cd.get('U1_std', [0]))
            nd['U2_mean'].append(cd.get('U2_mean', [0]))
            nd['U2_std'].append(cd.get('U2_std', [0]))
            nd['lr1to1'].append(cd['lr1to1'])
            nd['lr2to2'].append(cd['lr2to2'])
            nd['lr1to2'].append(cd['lr1to2'])
            nd['lr2to1'].append(cd['lr2to1'])

        # compute mean
        for k in nd:
            if k in ['lr1to1', 'lr2to2', 'lr1to2', 'lr2to1']:
                continue
            nd[k] = np.mean(np.stack(nd[k]), axis=0)

        return nd


@torch.no_grad()
def measure_pred_matrix_effect(clf, Ub_pred, Ub, Wb, class_idx):
    Xb_rec = torch.einsum('nk,kd->nd', Ub.cuda(), Wb.cuda())
    Yb_targ_preds = clf(Xb_rec)
    Yb_targ_probs = torch.softmax(Yb_targ_preds, dim=-1)
    Yb_targ_logprobs = torch.log_softmax(Yb_targ_preds, dim=-1)
    Ub_num_concepts = Ub.shape[1]
    num_classes = Yb_targ_probs.shape[-1]
    # lr, _ = regression_list[i]
    # Ub_pred = torch.FloatTensor(lr.predict(Ua[:, [i]]))
    # replace each column of U1 separately

    Xb_pred_rec = torch.einsum('nk,kd->nd', Ub_pred.cuda(), Wb.cuda())
    l2_dist = torch.norm(Xb_pred_rec - Xb_rec, dim=-1, p=2).cpu().type(
        torch.float16)

    Yb_replaced_preds = clf(Xb_pred_rec)
    Yb_replaced_logprobs = torch.log_softmax(Yb_replaced_preds, dim=-1)
    Yb_replaced_probs = torch.softmax(Yb_replaced_preds, dim=-1)
    delta_prob_target_class = Yb_replaced_probs[:, class_idx] - Yb_targ_probs[:, class_idx]
    kl_div = F.kl_div(Yb_replaced_logprobs, Yb_targ_logprobs, reduction='none', log_target=True).nansum(-1).cpu().type(torch.float16)
    matches = Yb_replaced_preds.argmax(-1) == Yb_targ_probs.argmax(-1)
    matches = matches.cpu().type(torch.float16)
    return dict(l2=l2_dist, kl=kl_div, match_acc=matches, dptc=delta_prob_target_class,
                replaced_probs=Yb_replaced_probs.cpu(), target_probs=Yb_targ_probs.cpu())

@torch.no_grad()
def run_replacement_test(clf, Ub_pred, Ub, Wb, class_idx):
    Xb_rec = torch.einsum('nk,kd->nd', Ub.cuda(), Wb.cuda())
    Yb_targ_preds = clf(Xb_rec)
    Yb_targ_probs = torch.softmax(Yb_targ_preds, dim=-1)
    Yb_targ_logprobs = torch.log_softmax(Yb_targ_preds, dim=-1)
    Ub_num_concepts = Ub.shape[1]
    num_classes = Yb_targ_probs.shape[-1]
    # lr, _ = regression_list[i]
    # Ub_pred = torch.FloatTensor(lr.predict(Ua[:, [i]]))
    # replace each column of U1 separately

    Ub_replaced = Ub.clone().cuda().unsqueeze(0).repeat(Ub_num_concepts, 1, 1)
    for j in range(Ub_num_concepts):
        Ub_replaced[j, :, j] = Ub_pred[:, j] * 0.5 + Ub[:, j] * 0.5

    Xb_replaced = torch.einsum('rnk,kd->rnd', Ub_replaced.cuda(), Wb.cuda())
    l2_dist = torch.norm(Xb_replaced - Xb_rec.unsqueeze(0).repeat(Ub_num_concepts, 1, 1), dim=-1, p=2).cpu().type(
        torch.float16)
    # for j in range(Ub_num_concepts):
    #     print(pearsonr(Ub_pred[:, j], Ub[:, j]).statistic)
    # for j in range(Ub_num_concepts):
    #     print(l2_dist.mean(1)[j].item())
    Yb_replaced_preds = clf(Xb_replaced.view(-1, Xb_replaced.shape[-1])).view(Xb_replaced.shape[0],
                                                                              Xb_replaced.shape[1], -1)
    Yb_replaced_logprobs = torch.log_softmax(Yb_replaced_preds, dim=-1)
    Yb_replaced_probs = torch.softmax(Yb_replaced_preds, dim=-1)
    delta_prob_target_class = Yb_replaced_probs[:, :, class_idx] - Yb_targ_probs[:, class_idx]
    kl_div = F.kl_div(Yb_replaced_logprobs.view(-1, num_classes),
                      Yb_targ_logprobs.unsqueeze(0).repeat(Ub_num_concepts, 1, 1).view(-1, num_classes),
                      reduction='none', log_target=True).nansum(-1).reshape(Ub_num_concepts, -1).cpu().type(
        torch.float16)
    matches = Yb_replaced_preds.argmax(-1) == Yb_targ_probs.argmax(-1)
    matches = matches.cpu().type(torch.float16)
    return dict(l2=l2_dist, kl=kl_div, match_acc=matches, dptc=delta_prob_target_class,
                replaced_probs=Yb_replaced_probs.cpu(), target_probs=Yb_targ_probs.cpu())


def enforce_one_sample_per_image(arr, image_list):
    # filter out samples from same image
    num_images = len(np.unique(image_list))
    num_concepts = arr.shape[1]
    arr_filtered_max = np.zeros((num_images, num_concepts), dtype=int)
    arr_filtered_min = np.zeros((num_images, num_concepts), dtype=int)
    for i in range(num_concepts):
        seen_list_max = set()
        seen_list_min = set()
        ind_max = 0
        ind_min = 0
        for j in range(arr.shape[0]):
            pidx_max = arr[-(j + 1), i]
            if image_list[pidx_max] not in seen_list_max:
                arr_filtered_max[ind_max, i] = pidx_max
                seen_list_max.add(image_list[pidx_max])
                ind_max += 1
            pidx_min = arr[j, i]
            if image_list[pidx_min] not in seen_list_min:
                arr_filtered_min[ind_min, i] = pidx_min
                seen_list_min.add(image_list[pidx_min])
                ind_min += 1

    return arr_filtered_max, arr_filtered_min

def unnormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x * std + mean


def visualize_concept_differences(args, plot_params, show=True, save=False, sim_pair_output_folder='concept_differences'):
    class_idx = plot_params['class_idx']
    method_dict = plot_params['method_dict']
    replacement_scores = plot_params['replacement_scores']
    scores = plot_params['scores']
    gt_labels = plot_params['gt_labels']
    U = plot_params['U']
    pred_U_unnorm = plot_params['pred_U_unnorm']
    image_list = plot_params['image_list']
    image_samples = plot_params['image_samples']
    # int_grad = plot_params['int_grad']
    class_name_func = plot_params['class_name_func']
    direction = plot_params["direction"]
    rep_scores = replacement_scores[direction]
    method_output_folder = method_dict['method_output_folder']
    concept_list = plot_params['concept_list']
    model_eval = plot_params['model_eval']
    folder = os.path.join(method_output_folder.split('concept_comparison/')[-1])
    mode = plot_params['mode']
    sim_pair_output_folder += f'_{mode}'
    viz_folder = os.path.join(args.visualization_output_root, 'outputs', 'visualizations',
                              f'{sim_pair_output_folder}', folder)
    os.makedirs(viz_folder, exist_ok=True)

    rep_score_filter = rep_scores['dptc']
    max_change = rep_score_filter.argsort(descending=True, dim=1).cpu()

    gt_labels_t = torch.tensor(gt_labels)

    err = np.abs((U - pred_U_unnorm))
    sign = (U - pred_U_unnorm) < 0
    err[sign] = -err[sign]
    top_d = err.argsort(axis=0)

    print(viz_folder)
    show = False
    save = True

    # rank_diff = mse
    U_rank = rankdata(U, axis=0, method='average')
    pred_U_rank = rankdata(pred_U_unnorm, axis=0, method='average')
    rank_diff = U_rank - pred_U_rank
    # mask = U < 0.05
    # rank_diff[mask] = 0
    # top_d = rank_diff.argsort(axis=0)

    _, topk_opi = enforce_one_sample_per_image((-U).argsort(axis=0), image_list)
    arr_filtered_min, arr_filtered_max = enforce_one_sample_per_image(top_d, image_list)
    arr = np.concatenate([arr_filtered_min, arr_filtered_max[::-1]], axis=0)
    top_d = arr
    f1_str = f'F1: m0: {model_eval[0]["stats"][str(class_idx)]["f1"]:0.2f} -- m1: {model_eval[1]["stats"][str(class_idx)]["f1"]:0.2f}'

    def make_image_grid(sel_images):
        if mode == '2x5':
            fig = plt.figure(figsize=(14.5, 5.95))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                             axes_pad=0.05,  # pad between Axes in inch.
                             )
        elif mode == '3x3':
            fig = plt.figure(figsize=(9., 9.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(3, 3),  # creates 2x2 grid of Axes
                             axes_pad=0.1,  # pad between Axes in inch.
                             )

        for ax, im in zip(grid, sel_images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            make_axes_invisible(ax)
        plt.tight_layout()
        return fig

    print(scores["pearsonr"][direction])
    concept_color = '#009E73'
    under_color = '#0072B2'
    over_color = '#D55E00'
    for i in concept_list:
        print()
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        axi = i
        ax = axes
        topk = U[:, axi].argsort()[::-1][:10]
        topk = topk_opi[:, axi][:10]
        top_d_max_inds = top_d[-10:, axi]
        # top_d_min_inds = top_d[:10, axi]
        min_real = U[topk, axi].min()
        top_d_min_inds = top_d[U[top_d[:, axi], axi] < min_real, axi][:10]

        ax.scatter(U[:, axi], pred_U_unnorm[:, axi], alpha=0.3, color='dimgrey', s=18)
        ax.scatter(U[top_d_max_inds, axi], pred_U_unnorm[top_d_max_inds, axi], color=over_color, s=26)
        ax.scatter(U[top_d_min_inds, axi], pred_U_unnorm[top_d_min_inds, axi], color=under_color, s=26)
        ax.scatter(U[topk, axi], pred_U_unnorm[topk, axi], color=concept_color, s=26, )
        ax.set_xlabel('Real Coeffs', fontsize=14)
        ax.set_ylabel('Predicted Coeffs', fontsize=14)
        # ax.scatter(U[gt_labels_t != class_idx, axi], pred_U_unnorm[gt_labels_t != class_idx, axi], color='r', marker='x')
        ax.set_title(
            f'Sim: {scores["pearsonr"][direction][axi]:0.2f} | KL-Div: {scores["replacement_kl"][direction][axi]:0.4f}')
        # plt.suptitle(f'{class_name_func(class_idx)}')
        ylim = axes.get_ylim()
        xlim = axes.get_xlim()
        # y = x line
        x = np.linspace(max(xlim[0], ylim[0]), min(ylim[1], xlim[1]), 100)
        y = x  # Since y = x

        # Create plot
        plt.plot(x, y, '--', color='red')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(viz_folder, f'{direction}_{class_idx}-{i}_scatter.png'), dpi=400)
            # plt.savefig(os.path.join(viz_folder, f'{direction}_{class_idx}-{i}_scatter.svg'))
        if show:
            plt.show()

        linewidth = 19
        sel_images = image_samples[top_d[-10:, i]].permute(0, 2, 3, 1).cpu().numpy()
        fig = make_image_grid(sel_images)
        fig.patch.set_linewidth(linewidth)
        fig.patch.set_edgecolor(over_color)
        if save:
            plt.savefig(os.path.join(viz_folder, f'{direction}_{class_idx}-{i}_over.png'), dpi=400)
        if show:
            plt.show()

        sel_images = image_samples[top_d_min_inds[:10]].permute(0, 2, 3, 1).cpu().numpy()
        fig = make_image_grid(sel_images)
        fig.patch.set_linewidth(linewidth)
        fig.patch.set_edgecolor(under_color)
        if save:
            plt.savefig(os.path.join(viz_folder, f'{direction}_{class_idx}-{i}_under.png'), dpi=400)
        if show:
            plt.show()

        topk = topk_opi[:, i][:10]
        sel_images = image_samples[topk].permute(0, 2, 3, 1).cpu().numpy()
        fig = make_image_grid(sel_images)
        fig.patch.set_linewidth(linewidth)
        fig.patch.set_edgecolor(concept_color)
        if save:
            plt.savefig(os.path.join(viz_folder, f'{direction}_{class_idx}-{i}_topk.png'), dpi=400)
        if show:
            plt.show()

    plt.close('all')

def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--visualization_output_root', type=str, default='./')
    parser.add_argument('--importance_output_root', type=str, default='./')
    parser.add_argument('--eval_dataset', type=str, default='imagenet')
    parser.add_argument('--data_split', type=str, default='val')
    parser.add_argument('--patchify', action='store_true')
    parser.add_argument('--visualize_concepts', action='store_true')
    parser.add_argument('--skip_saving', action='store_true')
    parser.add_argument('--sim_pair_output_folder', type=str, default='paper_qualitative')
    parser.add_argument('--selected_samples_json', type=str, default=None, required=True)
    parser.add_argument('--mode', type=str, default='3x3')

    args = parser.parse_args()

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
    class_list = param_dicts1['class_list']
    class_list_values = param_dicts1['class_list_values']

    with open(args.selected_samples_json, 'r') as f:
        selected_samples = json.load(f)

    class_list = []
    class_list_values = []
    for class_ in selected_samples['1to2']:
        class_list.append(int(class_))
        class_list_values.append(1)
    if '2to1' in selected_samples:
        for class_ in selected_samples['2to1']:
            class_list.append(int(class_))
            class_list_values.append(-1)

    print()

    concepts_folders = out['concepts_folders']
    activations_folder_0 = os.path.join(save_names1['activations_dir'], 'activations')
    activations_folder_1 = os.path.join(save_names2['activations_dir'], 'activations')

    # model0_int_grad_out_dir, model1_int_grad_out_dir = build_importance_output_dir(args, config, save_names1, save_names2, data_group_name)
    igs = args.cross_model_image_group_strategy
    dataset_name = args.dataset_0

    fe_outs = []
    models = []
    transforms = []
    clfs = []
    model_eval = {}
    model_save_names = [save_names1['model_name'], save_names2['model_name']]
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
        model = model_out['model']

        transform = model_out['test_transform'] if transform_type == 'test' else model_out['transform']
        transforms.append(transform)
        fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
        fe_outs.append(fe_out)

        backbone, fc = split_model(model)
        clfs.append(fc)

        # overwrite original num images (need original for accurately loading paths)
        # param_dicts['num_images'] = args.cmigs_num_images
        param_dicts['dataset_params']['num_images'] = num_images
        models.append(model)

        dataset_params = param_dicts['dataset_params']
        dataset = dataset_params['dataset_name']
        split = dataset_params['split']
        model_save_name = model_save_names[mi]
        path = f"model_evaluation/{dataset_name}/{model_name}_probs_{args.data_split}.pth"
        print(path)
        # with open(f'model_evaluation/{dataset_name}/{model_name}_{args.data_split}.json', 'r') as f:
        #     eval_dict = json.load(f)
        stats = eval_model.load_or_run_evaluation(f'model_evaluation/{dataset_name}/{model_name}_stats_{args.data_split}.json',
                                                  dataset, split, model_name, ckpt_path, f'./data')

        # with open(f'model_evaluation/{dataset_name}/{model_name}_stats_{args.data_split}.json', 'r') as f:
        #     stats = json.load(f)

        print(f"Model: {model_save_name}")
        # img_paths = np.array(list(eval_dict['predictions'].keys()))
        # labels = np.array(eval_dict['labels'])
        model_eval[mi] = {'mean': [], 'sample': [], 'stats': stats}

    act_hook1 = ActivationHook()
    act_hook1.register_hooks(fe_outs[0]['layer_names'][-1:], fe_outs[0]['layers'][-1:],
                             fe_outs[0]['post_activation_func'])

    act_hook2 = ActivationHook()
    act_hook2.register_hooks(fe_outs[1]['layer_names'][-1:], fe_outs[1]['layers'][-1:],
                             fe_outs[1]['post_activation_func'])

    layer_name_0 = fe_outs[0]['layer_names'][-1]
    layer_name_1 = fe_outs[1]['layer_names'][-1]
    dataset_name = param_dicts['dataset_params']['dataset_name']
    if dataset_name == 'imagenet':
        class_names_dict = imagenet_classes
        class_name_func = lambda x: class_names_dict[x].split(',')[0]
    elif dataset_name == 'nabirds' or dataset_name == 'nabirds_modified':
        class_names_dict = nabirds_classes
        class_name_func = lambda x: class_names_dict[x][1]
    elif dataset_name == 'nabirds_stanford_cars':
        class_names_dict = nabirds_classes
        num_nabirds_classes = len(nabirds_classes)
        for k, v in stanford_cars_classes.items():
            class_names_dict[k + num_nabirds_classes] = [-1, v]
        class_name_func = lambda x: class_names_dict[x][1]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_root_name = 'nabirds' if dataset_name == 'nabirds_modified' or dataset_name == 'nabirds_stanford_cars' else dataset_name
    unnormalized_test_transform = torchvision.transforms.Compose(model_out['test_transform'].transforms[:-1])
    normalize = model_out['test_transform'].transforms[-1]
    transform = torchvision.transforms.Compose(transforms[0].transforms[1:])
    print(igs)
    param_dicts1['dataset_params']['split'] = args.data_split
    param_dicts2['dataset_params']['split'] = args.data_split
    image_group, eval_dicts = create_image_group(strategy=igs, param_dicts=[param_dicts1, param_dicts2], return_eval_dict=True)
    image_paths = np.array(list(eval_dicts[0]['predictions'].keys()))
    pred_dict0 = eval_dicts[0]['predictions']
    pred_dict1 = eval_dicts[1]['predictions']
    label_dict = dict(zip(image_paths, eval_dicts[0]['labels']))
    image_id_dict = dict(zip(image_paths, range(len(image_paths))))

    for method_dict in comparison_methods:
        if method_dict['method'] == 'lasso_regression' or method_dict['method'] == 'lasso_regression_c':
            break

    # int_grad1_list = []
    # int_grad0_list = []
    # for class_idx in class_list:
    #     int_grad0 = torch.load(os.path.join(model0_int_grad_out_dir, f'{class_idx}.pth'))
    #     int_grad1 = torch.load(os.path.join(model1_int_grad_out_dir, f'{class_idx}.pth'))
    #
    #     num_concepts_0 = int_grad0.shape[1]
    #     num_concepts_1 = int_grad1.shape[1]
    #     int_grad0 = -1 * int_grad0
    #     int_grad1 = -1 * int_grad1
    #
    #     int_grad0_list.append(int_grad0)
    #     int_grad1_list.append(int_grad1)

    # with open(os.path.join(importance_output_dir, 'abs_max.json'), 'r') as f:
    #     abs_max = json.load(f)
    abs_max = {'0': 1, '1': 1}

    output_folder = 'regression_evaluation' + ('_patched' if args.patchify else '')
    method_output_folder = method_dict['method_output_folder']

    pbar = tqdm(class_list)
    ci = 0
    folder = os.path.join(method_output_folder.split('concept_comparison/')[-1])
    for class_idx in pbar:
        pbar.set_description(f'Class {class_idx}')
        try:
            comparison_file = os.path.join(method_output_folder, f'{class_idx}', f'{layer_name_0}-{layer_name_1}.pkl')
            with open(comparison_file, 'rb') as f:
                comparison_dict = pkl.load(f)
            if comparison_dict is None:
                continue
        except FileNotFoundError:
            print(f'No comparison file for class {class_idx}')
            continue
        if class_idx not in image_group or len(image_group[class_idx]) < 10:
            print(f'No images for class {class_idx}')
            continue
        comparison_dict = process_comparison_dict(comparison_dict)

        analysis_data = {}
        if not args.patchify:
            image_list = image_group[class_idx] * 4
            # off_classes = np.random.choice(list(image_group.keys()), 6)
            # for oc in off_classes:
            #     image_list += image_group[oc]

            out = ceh.select_class_and_load_images(image_path_list=image_list,
                                                      data_root=f'./data/{data_root_name}/',
                                                      # transform=model_out['test_transform'])
                                                      transform=model_out['transform'])
            print(out['num_images'])
            image_size = out['image_size']
            images_preprocessed = out['images_preprocessed']
            gt_labels = np.array([label_dict[path] for path in image_list])
            image_samples = unnormalize(images_preprocessed)
        else:
            image_list = image_group[class_idx]
            out = ceh.select_class_and_load_images(image_path_list=image_list,
                                                   data_root=f'./data/{data_root_name}/',
                                                   transform=unnormalized_test_transform)
            image_size = out['image_size']
            patch_size = param_dicts['feature_extraction_params']['patch_size']
            patches = ceh.patchify_images(out['images_preprocessed'], patch_size, strides=None)
            images_preprocessed = normalize(patches)
            ppi = patches.shape[0] // len(image_group[class_idx])
            image_list = np.array(image_group[class_idx])
            image_list = np.array(image_list).repeat(ppi)
            m0_image_preds = np.array([pred_dict0[path] for path in image_list])
            m1_image_preds = np.array([pred_dict1[path] for path in image_list])
            gt_labels = np.array([label_dict[path] for path in image_list])
            image_samples = patches

        m0_act = _batch_inference(models[0], images_preprocessed, batch_size=64,
                             resize=image_size,
                             device=param_dicts1['device'])
        act_hook1.concatenate_layer_activations()

        m1_act = _batch_inference(models[1], images_preprocessed, batch_size=64,
                             resize=image_size,
                             device=param_dicts2['device'])
        m0_preds = clfs[0](m0_act.cuda()).argmax(-1).cpu().numpy()
        m1_preds = clfs[1](m1_act.cuda()).argmax(-1).cpu().numpy()

        act_hook2.concatenate_layer_activations()

        activations0 = act_hook1.layer_activations[layer_name_0]
        activations1 = act_hook2.layer_activations[layer_name_1]

        act_hook1.reset_activation_dict()
        act_hook2.reset_activation_dict()

        concepts_0 = load_concepts(concepts_folders[0], layer_name_0, class_idx)
        concepts_1 = load_concepts(concepts_folders[1], layer_name_1, class_idx)
        W0 = concepts_0['W']
        U0 = compute_concept_coefficients(activations0, W0, method='fnnls')
        W1 = concepts_1['W']
        U1 = compute_concept_coefficients(activations1, W1, method='fnnls')

        m0_pred_recon = clfs[0](torch.FloatTensor((U0@W0)).cuda()).argmax(-1).cpu().numpy()
        m1_pred_recon = clfs[1](torch.FloatTensor((U1@W1)).cuda()).argmax(-1).cpu().numpy()
        acc_m0 = (m0_preds == gt_labels).mean()
        acc_m1 = (m1_preds == gt_labels).mean()
        acc_m0_recon = (m0_pred_recon == gt_labels).mean()
        acc_m1_recon = (m1_pred_recon == gt_labels).mean()

        gt_mask = gt_labels == class_idx
        m0_mask = m0_preds == class_idx
        m1_mask = m1_preds == class_idx

        def comp_prec_recall(gt_mask, mask):
            tp = (gt_mask & mask).sum()
            fp = (~gt_mask & mask).sum()
            fn = (gt_mask & ~mask).sum()
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            return prec, rec

        print('Acc M0: ', acc_m0)
        print('Acc M1: ', acc_m1)
        print('Acc M0(recon): ', acc_m0_recon)
        print('Acc M1(recon): ', acc_m1_recon)
        analysis_data['acc_m0'] = acc_m0
        analysis_data['acc_m1'] = acc_m1
        analysis_data['acc_m0_recon'] = acc_m0_recon
        analysis_data['acc_m1_recon'] = acc_m1_recon
        prec0, rec0 = comp_prec_recall(gt_mask, m0_mask)
        prec1, rec1 = comp_prec_recall(gt_mask, m1_mask)
        print('Prec 0: ', prec0, 'Rec 0: ', rec0)
        print('Prec 1: ', prec1, 'Rec 1: ', rec1)
        # if (m1_pred_recon == gt_labels).mean() > (m0_pred_recon == gt_labels).mean():
        #     print('Model 1 is better')
        # else:
        #     continue

        act0_mean, act0_std = comparison_dict['act1_mean'], comparison_dict['act1_std']
        act1_mean, act1_std = comparison_dict['act2_mean'], comparison_dict['act2_std']
        U0_mean, U0_std = comparison_dict['U1_mean'], comparison_dict['U1_std']
        U1_mean, U1_std = comparison_dict['U2_mean'], comparison_dict['U2_std']

        activations0_norm, _, _ = standardize(activations0, mean=act0_mean, std=act0_std)
        activations1_norm, _, _ = standardize(activations1, mean=act1_mean, std=act1_std)
        U0_norm, _, _ = standardize(U0, mean=U0_mean, std=U0_std)
        U1_norm, _, _ = standardize(U1, mean=U1_mean, std=U1_std)

        if type(comparison_dict['lr1to1']) == list:
            pred_U0 = []
            pred_U1 = []
            pred_U0U0 = []
            pred_U1U1 = []
            for lr_ind in range(len(comparison_dict['lr1to1'])):
                pred_U1.append(comparison_dict['lr1to2'][lr_ind].predict(activations0_norm))
                pred_U0.append(comparison_dict['lr2to1'][lr_ind].predict(activations1_norm))
                pred_U0U0.append(comparison_dict['lr1to1'][lr_ind].predict(activations0_norm))
                pred_U1U1.append(comparison_dict['lr2to2'][lr_ind].predict(activations1_norm))

            pred_U1 = np.stack(pred_U1, axis=0).mean(axis=0)
            pred_U0 = np.stack(pred_U0, axis=0).mean(axis=0)
            pred_U0U0 = np.stack(pred_U0U0, axis=0).mean(axis=0)
            pred_U1U1 = np.stack(pred_U1U1, axis=0).mean(axis=0)
        else:
            pred_U1 = comparison_dict['lr1to2'].predict(activations0_norm)
            pred_U0 = comparison_dict['lr2to1'].predict(activations1_norm)

            pred_U0U0 = comparison_dict['lr1to1'].predict(activations0_norm)
            pred_U1U1 = comparison_dict['lr2to2'].predict(activations1_norm)

        pred_U0_unnorm = unstandardize(pred_U0, mean=U0_mean, std=U0_std)
        pred_U1_unnorm = unstandardize(pred_U1, mean=U1_mean, std=U1_std)
        pred_U0U0_unnorm = unstandardize(pred_U0U0, mean=U0_mean, std=U0_std)
        pred_U1U1_unnorm = unstandardize(pred_U1U1, mean=U1_mean, std=U1_std)
        _tt = lambda x: torch.FloatTensor(x)
        # rep_out = run_replacement_test(clfs[1], _tt(pred_U1), _tt(U1), _tt(W1), class_idx)
        # compute similarity between real and predicted concept coefficients
        scores = {}
        for func_tuple in [('pearsonr', pearsonr), ('spearmanr', spearmanr), ('replacement', run_replacement_test)]:
            func_name, func = func_tuple
            if func_name == 'replacement':

                m0_pred_recon = clfs[0](torch.FloatTensor((pred_U0_unnorm @ W0)).cuda()).argmax(-1).cpu().numpy()
                m1_pred_recon = clfs[1](torch.FloatTensor((pred_U1_unnorm @ W1)).cuda()).argmax(-1).cpu().numpy()
                acc_m1to0_recon = (m0_pred_recon == gt_labels).mean()
                acc_m0to1_recon = (m1_pred_recon == gt_labels).mean()

                prec_0_regr_recon, rec_0_regr_recon = comp_prec_recall(gt_mask, m0_pred_recon == class_idx)
                prec_1_regr_recon, rec_1_regr_recon = comp_prec_recall(gt_mask, m1_pred_recon == class_idx)
                print('Prec 1->0 recon: ', prec_0_regr_recon, 'Rec 1->0 recon: ', rec_0_regr_recon)
                print('Prec 0->1 recon: ', prec_1_regr_recon, 'Rec 0->1 recon: ', rec_1_regr_recon)
                print('Acc M1->0: ', acc_m1to0_recon, 'Acc M0->1: ', acc_m0to1_recon)
                print('Acc M0(recon): ', acc_m0_recon, 'Acc M1(recon): ', acc_m1_recon)
                rep_out0 = run_replacement_test(clfs[0], _tt(pred_U0_unnorm), _tt(U0), _tt(W0), class_idx)
                rep_out1 = run_replacement_test(clfs[1], _tt(pred_U1_unnorm), _tt(U1), _tt(W1), class_idx)
                rep_out00 = run_replacement_test(clfs[0], _tt(pred_U0U0_unnorm), _tt(U0), _tt(W0), class_idx)
                rep_out11 = run_replacement_test(clfs[1], _tt(pred_U1U1_unnorm), _tt(U1), _tt(W1), class_idx)
                _scores = {}
                for key in ['l2', 'kl', 'match_acc', 'dptc']:
                    __scores = {'1to2': [], '2to1': [], '1to1': [], '2to2': []}
                    _key = f'replacement_{key}'
                    _scores[_key] = __scores

                    for i in range(U0.shape[1]):
                        _scores[_key]['2to2'].append(rep_out11[key].mean(1)[i].item())
                        _scores[_key]['1to1'].append(rep_out00[key].mean(1)[i].item())
                        _scores[_key]['2to1'].append(rep_out0[key].mean(1)[i].item())
                        _scores[_key]['1to2'].append(rep_out1[key].mean(1)[i].item())
                scores.update(_scores)
            elif func_name == 'matrix_effect':
                rep_out0 = measure_pred_matrix_effect(clfs[0], _tt(pred_U0_unnorm), _tt(U0), _tt(W0), class_idx)
                rep_out1 = measure_pred_matrix_effect(clfs[1], _tt(pred_U1_unnorm), _tt(U1), _tt(W1), class_idx)
                rep_out00 = measure_pred_matrix_effect(clfs[0], _tt(pred_U0U0_unnorm), _tt(U0), _tt(W0), class_idx)
                rep_out11 = measure_pred_matrix_effect(clfs[1], _tt(pred_U1U1_unnorm), _tt(U1), _tt(W1), class_idx)
                _scores = {}
                for key in ['l2', 'kl', 'match_acc', 'dptc']:
                    __scores = {'1to2': [], '2to1': [], '1to1': [], '2to2': []}
                    _key = f'me_{key}'
                    _scores[_key] = __scores
                    _scores[_key]['2to2'].append(rep_out11[key].mean(-1).item())
                    _scores[_key]['1to1'].append(rep_out00[key].mean(-1).item())
                    _scores[_key]['2to1'].append(rep_out0[key].mean(-1).item())
                    _scores[_key]['1to2'].append(rep_out1[key].mean(-1).item())
                scores.update(_scores)
            else:
                _scores = {'1to2_norm': [], '2to1_norm': [], '1to1_norm': [], '2to2_norm': [], '1to2': [], '2to1': [], '1to1': [], '2to2': []}
                for i in range(U0.shape[1]):
                    # _scores['1to1_norm'].append(func(U0_norm[:, i], pred_U0U0[:, i]).statistic)
                    # _scores['2to2_norm'].append(func(U1_norm[:, i], pred_U1U1[:, i]).statistic)
                    # _scores['2to1_norm'].append(func(U0_norm[:, i], pred_U0[:, i]).statistic)
                    # _scores['1to2_norm'].append(func(U1_norm[:, i], pred_U1[:, i]).statistic)

                    _scores['1to1'].append(func(U0[:, i], pred_U0U0_unnorm[:, i]).statistic)
                    _scores['2to2'].append(func(U1[:, i], pred_U1U1_unnorm[:, i]).statistic)
                    _scores['2to1'].append(func(U0[:, i], pred_U0_unnorm[:, i]).statistic)
                    _scores['1to2'].append(func(U1[:, i], pred_U1_unnorm[:, i]).statistic)
                scores[func_name] = _scores

        comparison_output_file = comparison_file.replace('concept_comparison', output_folder)
        os.makedirs(os.path.dirname(comparison_output_file), exist_ok=True)
        with open(comparison_output_file, 'rb') as f:
            comp_dict = pkl.load(f)

        if args.visualize_concepts:
            replacement_scores = {'1to2': rep_out0, '2to1': rep_out1, '1to1': rep_out00, '2to2': rep_out11}
            if class_list_values[ci] <= 0:
                plot_params = {
                    'class_idx': class_idx,
                    'concept_list': selected_samples['2to1'][str(class_idx)],
                    'class_name_func': class_name_func,
                    'image_samples': image_samples,
                    'gt_labels': gt_labels,
                    'm0_preds': m0_preds,
                    'm1_preds': m1_preds,
                    'U': U0,
                    'pred_U_unnorm': pred_U0_unnorm,
                    'scores': scores,
                    'replacement_scores': replacement_scores,
                    'image_list': image_list,
                    # 'int_grad': int_grad0_list[ci] / abs_max['0'],
                    'method_dict': method_dict,
                    'direction': "2to1",
                    'model_eval': model_eval,
                    'mode': args.mode
                }
                visualize_concept_differences(args, plot_params, save=True, show=False, sim_pair_output_folder=args.sim_pair_output_folder)
            if class_list_values[ci] >= 0:
                plot_params = {
                    'class_idx': class_idx,
                    'concept_list': selected_samples['1to2'][str(class_idx)],
                    'class_name_func': class_name_func,
                    'image_samples': image_samples,
                    'gt_labels': gt_labels,
                    'm0_preds': m0_preds,
                    'm1_preds': m1_preds,
                    'U': U1,
                    'pred_U_unnorm': pred_U1_unnorm,
                    'scores': scores,
                    'replacement_scores': replacement_scores,
                    'image_list': image_list,
                    # 'int_grad': int_grad1_list[ci] / abs_max['0'],
                    'method_dict': method_dict,
                    'direction': "1to2",
                    'model_eval': model_eval,
                    'mode': args.mode
                }
                visualize_concept_differences(args, plot_params, save=True, show=False, sim_pair_output_folder=args.sim_pair_output_folder)
        ci += 1


if __name__ == '__main__':

    main()
