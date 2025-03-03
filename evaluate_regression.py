import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHook
import json
import os
from tqdm import tqdm
import copy
from argparse import Namespace
from extract_model_activations import create_image_group, _batch_inference
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from scipy.stats import pearsonr, spearmanr, rankdata
# from concept_integrated_gradients import build_output_dir as build_importance_output_dir
import torchvision

from PIL import Image, ImageOps
from compare_models import standardize, unstandardize, build_model_comparison_parser, set_seed, build_output_dir, process_config
from src.utils.funcs import _batch_inference, correlation_comparison, load_concepts, compute_concept_coefficients
from src.utils.model_loader import split_model

import torch.nn.functional as F
import sklearn

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
            if cd['U1_mean'] is not None:
                nd['lr1to1_pearson'].append(cd['lr1to1_pearson'])
                nd['lr2to1_pearson'].append(cd['lr2to1_pearson'])
                nd['lr1to1_coef'].append(cd['lr1to1'].coef_)
                nd['lr2to1_coef'].append(cd['lr2to1'].coef_)
                nd['lr1to1_intercept'].append(cd['lr1to1'].intercept_)
                nd['lr2to1_intercept'].append(cd['lr2to1'].intercept_)
                nd['act1_mean'].append(cd.get('act1_mean', [0]))
                nd['act1_std'].append(cd.get('act1_std', [0]))
                nd['U1_mean'].append(cd.get('U1_mean', [0]))
                nd['U1_std'].append(cd.get('U1_std', [0]))
                nd['lr1to1'].append(cd['lr1to1'])
                nd['lr2to1'].append(cd['lr2to1'])
            if cd['U2_mean'] is not None:
                nd['lr2to2_pearson'].append(cd['lr2to2_pearson'])
                nd['lr1to2_pearson'].append(cd['lr1to2_pearson'])
                nd['lr2to2_coef'].append(cd['lr2to2'].coef_)
                nd['lr1to2_coef'].append(cd['lr1to2'].coef_)
                nd['lr2to2_intercept'].append(cd['lr2to2'].intercept_)
                nd['lr1to2_intercept'].append(cd['lr1to2'].intercept_)
                nd['act2_mean'].append(cd.get('act2_mean', [0]))
                nd['act2_std'].append(cd.get('act2_std', [0]))
                nd['U2_mean'].append(cd.get('U2_mean', [0]))
                nd['U2_std'].append(cd.get('U2_std', [0]))
                nd['lr2to2'].append(cd['lr2to2'])
                nd['lr1to2'].append(cd['lr1to2'])

        # compute mean
        for k in nd:
            if k in ['lr1to1', 'lr2to2', 'lr1to2', 'lr2to1']:
                continue
            if len(nd[k]) == 0:
                nd[k] = None
            else:
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
    mean = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
    std = torch.tensor([0.5000, 0.5000, 0.5000]).view(1, 3, 1, 1)
    return x * std + mean


def comp_prec_recall(gt_mask, mask):
    tp = (gt_mask & mask).sum()
    fp = (~gt_mask & mask).sum()
    fn = (gt_mask & ~mask).sum()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return prec, rec

def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--eval_dataset', type=str, default='imagenet')
    parser.add_argument('--data_split', type=str, default='val')
    parser.add_argument('--patchify', action='store_true')
    parser.add_argument('--skip_saving', action='store_true')

    args = parser.parse_args()

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    dataset = config['dataset']
    num_images = config['num_images']
    transform_type = config['transform_type']
    seed = config['seed']
    set_seed(seed)
    comparison_methods = config['methods']
    comparison_name = config['comparison_name']
    args.comparison_save_name = comparison_name

    method_output_dir = build_output_dir(args.comparison_output_root, 'concept_comparison', comparison_name)
    data_group_name, method_output_folders = process_config(config, method_output_dir)

    print(method_output_folders)

    out = build_model_comparison_param_dicts(args)
    param_dicts1 = out['param_dicts1']
    save_names1 = out['save_names1']
    param_dicts2 = out['param_dicts2']
    save_names2 = out['save_names2']
    model0_name = save_names1['model_name']
    model1_name = save_names2['model_name']
    class_list = param_dicts1['class_list']

    concepts_folders = out['concepts_folders']
    activations_folder_0 = os.path.join(save_names1['activations_dir'], 'activations')
    activations_folder_1 = os.path.join(save_names2['activations_dir'], 'activations')

    # importance_output_dir = build_importance_output_dir(args)
    # model0_int_grad_out_dir = os.path.join(importance_output_dir, save_names1['model_name'])
    # model1_int_grad_out_dir = os.path.join(importance_output_dir, save_names2['model_name'])
    igs = args.cross_model_image_group_strategy
    # dataset_name = args.dataset_0

    fe_outs = []
    models = []
    transforms = []
    clfs = []
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

    act_hook1 = ActivationHook()
    act_hook1.register_hooks(fe_outs[0]['layer_names'][-1:], fe_outs[0]['layers'][-1:],
                             fe_outs[0]['post_activation_func'])

    act_hook2 = ActivationHook()
    act_hook2.register_hooks(fe_outs[1]['layer_names'][-1:], fe_outs[1]['layers'][-1:],
                             fe_outs[1]['post_activation_func'])

    layer_name_0 = fe_outs[0]['layer_names'][-1]
    layer_name_1 = fe_outs[1]['layer_names'][-1]
    dataset_name = param_dicts['dataset_params']['dataset_name']

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

    for method_dict in comparison_methods:
        if method_dict['method'] == 'lasso_regression' or method_dict['method'] == 'lasso_regression_c':
            break

    output_folder = 'regression_evaluation' + ('_patched' if args.patchify else '')
    method_output_folder = method_dict['method_output_folder']

    pbar = tqdm(class_list)
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

            out = ceh.select_class_and_load_images_v2(image_path_list=image_list,
                                                      data_root=f'./data/{dataset}/',
                                                      # transform=model_out['test_transform'])
                                                      transform=model_out['transform'])
            print(out['num_images'])
            image_size = out['image_size']
            images_preprocessed = out['images_preprocessed']
            gt_labels = np.array([label_dict[path] for path in image_list])
            image_samples = unnormalize(images_preprocessed)
        else:
            out = ceh.select_class_and_load_images_v2(image_path_list=image_group[class_idx],
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
        gt_mask = gt_labels == class_idx
        if concepts_0 is not None:
            W0 = concepts_0['W']
            U0 = compute_concept_coefficients(activations0, W0, method='fnnls')
            m0_pred_recon = clfs[0](torch.FloatTensor((U0 @ W0)).cuda()).argmax(-1).cpu().numpy()
            acc_m0 = (m0_preds == gt_labels).mean()
            acc_m0_recon = (m0_pred_recon == gt_labels).mean()
            m0_mask = m0_preds == class_idx
            print('Acc M0: ', acc_m0)
            print('Acc M0(recon): ', acc_m0_recon)
            analysis_data['acc_m0'] = acc_m0
            analysis_data['acc_m0_recon'] = acc_m0_recon
            prec0, rec0 = comp_prec_recall(gt_mask, m0_mask)
            print('Prec 0: ', prec0, 'Rec 0: ', rec0)
            act0_mean, act0_std = comparison_dict['act1_mean'], comparison_dict['act1_std']
            U0_mean, U0_std = comparison_dict['U1_mean'], comparison_dict['U1_std']
            activations0_norm, _, _ = standardize(activations0, mean=act0_mean, std=act0_std)
            U0_norm, _, _ = standardize(U0, mean=U0_mean, std=U0_std)
        else:
            U0 = None
            acc_m0_recon = None

        if concepts_1 is not None:
            W1 = concepts_1['W']
            U1 = compute_concept_coefficients(activations1, W1, method='fnnls')
            m1_pred_recon = clfs[1](torch.FloatTensor((U1@W1)).cuda()).argmax(-1).cpu().numpy()
            acc_m1 = (m1_preds == gt_labels).mean()
            acc_m1_recon = (m1_pred_recon == gt_labels).mean()
            m1_mask = m1_preds == class_idx
            print('Acc M1: ', acc_m1)
            print('Acc M1(recon): ', acc_m1_recon)
            analysis_data['acc_m1'] = acc_m1
            analysis_data['acc_m1_recon'] = acc_m1_recon
            prec1, rec1 = comp_prec_recall(gt_mask, m1_mask)
            print('Prec 1: ', prec1, 'Rec 1: ', rec1)
            act1_mean, act1_std = comparison_dict['act2_mean'], comparison_dict['act2_std']
            U1_mean, U1_std = comparison_dict['U2_mean'], comparison_dict['U2_std']
            activations1_norm, _, _ = standardize(activations1, mean=act1_mean, std=act1_std)
            U1_norm, _, _ = standardize(U1, mean=U1_mean, std=U1_std)
        else:
            U1 = None
            acc_m1_recon = None

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

            if len(pred_U0) > 1:
                pred_U0 = np.stack(pred_U0, axis=0).mean(axis=0)
                pred_U0U0 = np.stack(pred_U0U0, axis=0).mean(axis=0)
            else:
                pred_U0 = None
                pred_U0U0 = None

            if len(pred_U1) > 1:
                pred_U1 = np.stack(pred_U1, axis=0).mean(axis=0)
                pred_U1U1 = np.stack(pred_U1U1, axis=0).mean(axis=0)
            else:
                pred_U1 = None
                pred_U1U1 = None
        else:
            pred_U1 = comparison_dict['lr1to2'].predict(activations0_norm)
            pred_U0 = comparison_dict['lr2to1'].predict(activations1_norm)

            pred_U0U0 = comparison_dict['lr1to1'].predict(activations0_norm)
            pred_U1U1 = comparison_dict['lr2to2'].predict(activations1_norm)

        if pred_U0 is not None:
            pred_U0_unnorm = unstandardize(pred_U0, mean=U0_mean, std=U0_std)
            pred_U0U0_unnorm = unstandardize(pred_U0U0, mean=U0_mean, std=U0_std)
        else:
            pred_U0_unnorm = None
            pred_U0U0_unnorm = None

        if pred_U1 is not None:
            pred_U1_unnorm = unstandardize(pred_U1, mean=U1_mean, std=U1_std)
            pred_U1U1_unnorm = unstandardize(pred_U1U1, mean=U1_mean, std=U1_std)
        else:
            pred_U1_unnorm = None
            pred_U1U1_unnorm = None

        _tt = lambda x: torch.FloatTensor(x)
        # rep_out = run_replacement_test(clfs[1], _tt(pred_U1), _tt(U1), _tt(W1), class_idx)
        # compute similarity between real and predicted concept coefficients
        scores = {}
        for func_tuple in [('pearsonr', pearsonr), ('spearmanr', spearmanr), ('replacement', run_replacement_test),
                           ('matrix_effect', measure_pred_matrix_effect)]:
            func_name, func = func_tuple
            if func_name == 'replacement':
                if pred_U0 is not None:
                    m0_pred_recon = clfs[0](torch.FloatTensor((pred_U0_unnorm @ W0)).cuda()).argmax(-1).cpu().numpy()
                    acc_m1to0_recon = (m0_pred_recon == gt_labels).mean()
                    prec_0_regr_recon, rec_0_regr_recon = comp_prec_recall(gt_mask, m0_pred_recon == class_idx)
                    print('Prec 1->0 recon: ', prec_0_regr_recon, 'Rec 1->0 recon: ', rec_0_regr_recon)
                    rep_out0 = run_replacement_test(clfs[0], _tt(pred_U0_unnorm), _tt(U0), _tt(W0), class_idx)
                    rep_out00 = run_replacement_test(clfs[0], _tt(pred_U0U0_unnorm), _tt(U0), _tt(W0), class_idx)
                else:
                    rep_out0 = None
                    rep_out00 = None
                    acc_m1to0_recon = None

                if pred_U1 is not None:
                    m1_pred_recon = clfs[1](torch.FloatTensor((pred_U1_unnorm @ W1)).cuda()).argmax(-1).cpu().numpy()
                    acc_m0to1_recon = (m1_pred_recon == gt_labels).mean()
                    prec_1_regr_recon, rec_1_regr_recon = comp_prec_recall(gt_mask, m1_pred_recon == class_idx)
                    print('Prec 0->1 recon: ', prec_1_regr_recon, 'Rec 0->1 recon: ', rec_1_regr_recon)
                    rep_out1 = run_replacement_test(clfs[1], _tt(pred_U1_unnorm), _tt(U1), _tt(W1), class_idx)
                    rep_out11 = run_replacement_test(clfs[1], _tt(pred_U1U1_unnorm), _tt(U1), _tt(W1), class_idx)
                else:
                    rep_out1 = None
                    rep_out11 = None
                    acc_m0to1_recon = None

                print('Acc M1->0: ', acc_m1to0_recon, 'Acc M0->1: ', acc_m0to1_recon)
                print('Acc M0(recon): ', acc_m0_recon, 'Acc M1(recon): ', acc_m1_recon)

                _scores = {}
                for key in ['l2', 'kl', 'match_acc', 'dptc']:
                    __scores = {'1to2': [], '2to1': [], '1to1': [], '2to2': []}
                    _key = f'replacement_{key}'
                    _scores[_key] = __scores

                    if pred_U0 is not None:
                        for i in range(U0.shape[1]):
                            _scores[_key]['1to1'].append(rep_out00[key].mean(1)[i].item())
                            _scores[_key]['2to1'].append(rep_out0[key].mean(1)[i].item())

                    if pred_U1 is not None:
                        for i in range(U1.shape[1]):
                            _scores[_key]['2to2'].append(rep_out11[key].mean(1)[i].item())
                            _scores[_key]['1to2'].append(rep_out1[key].mean(1)[i].item())

                scores.update(_scores)
            elif func_name == 'matrix_effect':
                if pred_U0 is not None:
                    rep_out0 = measure_pred_matrix_effect(clfs[0], _tt(pred_U0_unnorm), _tt(U0), _tt(W0), class_idx)
                    rep_out00 = measure_pred_matrix_effect(clfs[0], _tt(pred_U0U0_unnorm), _tt(U0), _tt(W0), class_idx)
                if pred_U1 is not None:
                    rep_out1 = measure_pred_matrix_effect(clfs[1], _tt(pred_U1_unnorm), _tt(U1), _tt(W1), class_idx)
                    rep_out11 = measure_pred_matrix_effect(clfs[1], _tt(pred_U1U1_unnorm), _tt(U1), _tt(W1), class_idx)

                _scores = {}
                for key in ['l2', 'kl', 'match_acc', 'dptc']:
                    __scores = {'1to2': [], '2to1': [], '1to1': [], '2to2': []}
                    _key = f'me_{key}'
                    _scores[_key] = __scores

                    if pred_U0 is not None:
                        _scores[_key]['1to1'].append(rep_out00[key].mean(-1).item())
                        _scores[_key]['2to1'].append(rep_out0[key].mean(-1).item())

                    if pred_U1 is not None:
                        _scores[_key]['2to2'].append(rep_out11[key].mean(-1).item())
                        _scores[_key]['1to2'].append(rep_out1[key].mean(-1).item())

                scores.update(_scores)
            else:
                _scores = {'1to2_norm': [], '2to1_norm': [], '1to1_norm': [], '2to2_norm': [], '1to2': [], '2to1': [], '1to1': [], '2to2': []}

                if pred_U0 is not None:
                    for i in range(U0.shape[1]):
                        _scores['1to1'].append(func(U0[:, i], pred_U0U0_unnorm[:, i]).statistic)
                        _scores['2to1'].append(func(U0[:, i], pred_U0_unnorm[:, i]).statistic)

                if pred_U1 is not None:
                    for i in range(U1.shape[1]):
                        _scores['2to2'].append(func(U1[:, i], pred_U1U1_unnorm[:, i]).statistic)
                        _scores['1to2'].append(func(U1[:, i], pred_U1_unnorm[:, i]).statistic)

                scores[func_name] = _scores

        comparison_output_file = comparison_file.replace('concept_comparison', output_folder)
        os.makedirs(os.path.dirname(comparison_output_file), exist_ok=True)
        if not args.skip_saving:
            with open(comparison_output_file, 'wb') as f:
                pkl.dump({'scores': scores, 'analysis_data': analysis_data}, f)



if __name__ == '__main__':
    '''
    --class_list_path class_lists/sample_list_r18_vs_50.json
    --class_list_path
concept_mining_class_lists/resnet50.a2_in1k_resnet18.a2_in1k_top20.json
    '''
    # main()
    print()
    # print()
    main()
