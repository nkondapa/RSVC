import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import torch
from math import ceil
from src.utils.parser_helper import concept_comparison_parser
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHook
import json
import os
from tqdm import tqdm
from extract_model_activations import create_image_group, _batch_inference
from src.utils.parser_helper import build_model_comparison_param_dicts
from src.utils.model_loader import split_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from scipy.stats import pearsonr, spearmanr
from scipy.stats._warnings_errors import ConstantInputWarning, NearConstantInputWarning
from sklearn.utils._testing import ignore_warnings
import pickle as pkl
import multiprocessing
import time
import random
from celer import Lasso as CelerLasso
from sklearn.model_selection import KFold
from src.utils.funcs import _batch_inference, correlation_comparison, load_concepts, compute_concept_coefficients


def build_output_dir(args, comparison_name):
    path = os.path.join(args.comparison_output_root, 'outputs/data/concept_comparison', comparison_name)
    os.makedirs(path, exist_ok=True)
    return path


def standardize(X, mean=None, std=None):
    if mean is not None:
        assert std is not None

    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8

    tmp = (X - mean) / std
    return tmp, mean, std


def unstandardize(X, mean, std):
    if mean is not None:
        assert std is not None

    return X * std + mean

# @ TODO clean up print statements
@ignore_warnings(category=ConstantInputWarning)
def regression_comparison(method_name, params, activations1, activations2, U1, U2):
    '''
    This test measures the ability of the network to decode the concept coefficients from the activation vector
    The test measures activations1 * x11 = U1, activations2 * x22 = U2, activations1 * x12 = U2, activations2 * x21 = U1
    :return:
    '''

    if method_name == 'linear_regression':
        method = LinearRegression
    elif method_name == 'ridge_regression':
        method = Ridge
    elif method_name == 'lasso_regression':
        method = Lasso
    elif method_name == 'lasso_regression_c':
        method = CelerLasso
    elif method_name == 'elastic_net':
        method = ElasticNet
    elif method_name == 'k_nearest_neighbors':
        method = KNeighborsRegressor
    elif method_name == 'radius_neighbors':
        method = RadiusNeighborsRegressor
    else:
        raise ValueError(f'Unknown regression method: {method_name}')

    # STANDARDIZE
    act1_mean, act1_std = None, None
    act2_mean, act2_std = None, None
    if params.get('standardize', False):
        activations1, act1_mean, act1_std = standardize(activations1)
        activations2, act2_mean, act2_std = standardize(activations2)

    U1_mean, U1_std = None, None
    U2_mean, U2_std = None, None
    if params.get('standardize_targets', False):
        if U1 is not None:
            U1, U1_mean, U1_std = standardize(U1)
        if U2 is not None:
            U2, U2_mean, U2_std = standardize(U2)

    rng = np.random.default_rng(params['seed'])
    if params['regression_train_pct'] is not None:
        indices = np.arange(activations1.shape[0])
        num_train = params['num_train']
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        splits = [(train_indices, test_indices)]
    elif params['num_folds'] is not None:
        splits = []
        for i in range(len(params['num_folds']['train_indices'])):
            train_inds = params['num_folds']['train_indices'][i]
            test_inds = params['num_folds']['test_indices'][i]
            rng.shuffle(train_inds)
            splits.append([train_inds, test_inds])
    else:
        raise ValueError('Must specify either regression_train_pct or num_folds')

    method_params = params['regression_params']

    outs = {}
    for k, (train_indices, test_indices) in enumerate(splits):
        # print(f'Fold {k}')
        # split data train and test
        train_act1 = activations1[train_indices]
        test_act1 = activations1[test_indices]
        train_act2 = activations2[train_indices]
        test_act2 = activations2[test_indices]

        if U1 is not None:
            train_U1 = U1[train_indices]
            test_U1 = U1[test_indices]
            lr1to1 = method(**method_params)
            lr2to1 = method(**method_params)
            lr1to1.fit(train_act1, train_U1)
            lr2to1.fit(train_act2, train_U1)
            pred_A1U1 = lr1to1.predict(test_act1)
            pred_A2U1 = lr2to1.predict(test_act2)
            # pearson correlation
            lr1to1_pearson = np.array([pearsonr(test_U1[:, i], pred_A1U1[:, i]).statistic for i in range(U1.shape[1])])
            lr2to1_pearson = np.array([pearsonr(test_U1[:, i], pred_A2U1[:, i]).statistic for i in range(U1.shape[1])])
            # score
            lr1to1_score = r2_score(test_U1, pred_A1U1, multioutput='raw_values')
            lr2to1_score = r2_score(test_U1, pred_A2U1, multioutput='raw_values')
        else:
            lr1to1 = None
            lr2to1 = None
            lr1to1_pearson = None
            lr2to1_pearson = None
            lr1to1_score = None
            lr2to1_score = None

        if U2 is not None:
            train_U2 = U2[train_indices]
            test_U2 = U2[test_indices]
            lr1to2 = method(**method_params)
            lr2to2 = method(**method_params)
            lr1to2.fit(train_act1, train_U2)
            lr2to2.fit(train_act2, train_U2)
            pred_A1U2 = lr1to2.predict(test_act1)
            pred_A2U2 = lr2to2.predict(test_act2)
            # pearson correlation
            lr1to2_pearson = np.array([pearsonr(test_U2[:, i], pred_A1U2[:, i]).statistic for i in range(U2.shape[1])])
            lr2to2_pearson = np.array([pearsonr(test_U2[:, i], pred_A2U2[:, i]).statistic for i in range(U2.shape[1])])
            # score
            lr1to2_score = r2_score(test_U2, pred_A1U2, multioutput='raw_values')
            lr2to2_score = r2_score(test_U2, pred_A2U2, multioutput='raw_values')
        else:
            lr1to2 = None
            lr2to2 = None
            lr1to2_pearson = None
            lr2to2_pearson = None
            lr1to2_score = None
            lr2to2_score = None

        nc_i = U1.shape[1] if U1 is not None else 0
        nc_j = U2.shape[1] if U2 is not None else 0
        outs[k] = {
            'act1_mean': act1_mean,
            'act1_std': act1_std,
            'act2_mean': act2_mean,
            'act2_std': act2_std,
            'U1_mean': U1_mean,
            'U1_std': U1_std,
            'U2_mean': U2_mean,
            'U2_std': U2_std,

            'lr1to1': lr1to1,
            'lr1to2': lr1to2,
            'lr2to1': lr2to1,
            'lr2to2': lr2to2,
            'metadata': {'method': method_name, 'num_concepts_i': nc_i, 'num_concepts_j': nc_j},

            'lr1to1_score': lr1to1_score,
            'lr1to2_score': lr1to2_score,
            'lr2to1_score': lr2to1_score,
            'lr2to2_score': lr2to2_score,

            'lr1to1_pearson': lr1to1_pearson,
            'lr1to2_pearson': lr1to2_pearson,
            'lr2to1_pearson': lr2to1_pearson,
            'lr2to2_pearson': lr2to2_pearson,
        }
        print()
        print(lr1to2_score)
        print(lr2to1_score)
        s1 = ''
        s2 = ''
        if U1 is not None:
            s1 += f'U1: {lr1to1_score.mean()}, {lr2to1_score.mean()}'
            s2 += f'U1: {lr1to1_pearson.mean()}, {lr2to1_pearson.mean()}'
        if U2 is not None:
            s1 += f'  U2: {lr1to2_score.mean()}, {lr2to2_score.mean()}'
            s2 += f'  U2: {lr1to2_pearson.mean()}, {lr2to2_pearson.mean()}'
        print(s1)
        print(s2)
        print()

    if len(outs) == 1:
        return outs[0]
    else:
        return outs


@ignore_warnings(category=ConstantInputWarning)
def concept_regression_comparison(method_name, params, U1, U2):
    '''
    This test measures the ability of the network to decode the concept coefficients from the activation vector
    The test measures activations1 * x11 = U1, activations2 * x22 = U2, activations1 * x12 = U2, activations2 * x21 = U1
    :return:
    '''

    if method_name == 'concept_linear_regression':
        method = LinearRegression
    elif method_name == 'concept_ridge_regression':
        method = Ridge
    elif method_name == 'concept_lasso_regression':
        method = Lasso
    elif method_name == 'concept_lasso_regression_c':
        method = CelerLasso
    elif method_name == 'concept_elastic_net':
        method = ElasticNet
    elif method_name == 'concept_k_nearest_neighbors':
        method = KNeighborsRegressor
    elif method_name == 'concept_radius_neighbors':
        method = RadiusNeighborsRegressor
    else:
        raise ValueError(f'Unknown regression method: {method_name}')


    U1_mean, U1_std = None, None
    U2_mean, U2_std = None, None
    if params.get('standardize_targets', False):
        U1, U1_mean, U1_std = standardize(U1)
        U2, U2_mean, U2_std = standardize(U2)

    rng = np.random.default_rng(params['seed'])
    if params['regression_train_pct'] is not None:
        indices = np.arange(U1.shape[0])
        num_train = params['num_train']
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        splits = [(train_indices, test_indices)]
    elif params['num_folds'] is not None:
        splits = []
        for i in range(len(params['num_folds']['train_indices'])):
            train_inds = params['num_folds']['train_indices'][i]
            test_inds = params['num_folds']['test_indices'][i]
            rng.shuffle(train_inds)
            splits.append([train_inds, test_inds])
    else:
        raise ValueError('Must specify either regression_train_pct or num_folds')

    method_params = params['regression_params']

    outs = {}
    for k, (train_indices, test_indices) in enumerate(splits):

        train_U1 = U1[train_indices]
        train_U2 = U2[train_indices]
        test_U1 = U1[test_indices]
        test_U2 = U2[test_indices]

        lr1to1 = method(**method_params)
        lr1to2 = method(**method_params)
        lr2to1 = method(**method_params)
        lr2to2 = method(**method_params)

        lr1to1.fit(train_U1, train_U1)
        lr1to2.fit(train_U1, train_U2)
        lr2to1.fit(train_U2, train_U1)
        lr2to2.fit(train_U2, train_U2)

        # train_pred_A1U1 = lr1to1.predict(train_U1)
        # train_pred_A1U2 = lr1to2.predict(train_U1)
        # train_pred_A2U2 = lr2to2.predict(train_U2)
        # train_pred_A2U1 = lr2to1.predict(train_U2)

        pred_A1U1 = lr1to1.predict(test_U1)
        pred_A1U2 = lr1to2.predict(test_U1)
        pred_A2U2 = lr2to2.predict(test_U2)
        pred_A2U1 = lr2to1.predict(test_U2)

        # pearson correlation
        lr1to1_pearson = np.array([pearsonr(test_U1[:, i], pred_A1U1[:, i]).statistic for i in range(U1.shape[1])])
        lr1to2_pearson = np.array([pearsonr(test_U2[:, i], pred_A1U2[:, i]).statistic for i in range(U2.shape[1])])
        lr2to1_pearson = np.array([pearsonr(test_U1[:, i], pred_A2U1[:, i]).statistic for i in range(U1.shape[1])])
        lr2to2_pearson = np.array([pearsonr(test_U2[:, i], pred_A2U2[:, i]).statistic for i in range(U2.shape[1])])

        # train_lr1to1_pearson = np.array([pearsonr(train_U1[:, i], train_pred_A1U1[:, i]).statistic for i in range(U1.shape[1])])
        # train_lr1to2_pearson = np.array([pearsonr(train_U2[:, i], train_pred_A1U2[:, i]).statistic for i in range(U2.shape[1])])
        # train_lr2to1_pearson = np.array([pearsonr(train_U1[:, i], train_pred_A2U1[:, i]).statistic for i in range(U1.shape[1])])
        # train_lr2to2_pearson = np.array([pearsonr(train_U2[:, i], train_pred_A2U2[:, i]).statistic for i in range(U2.shape[1])])

        # score
        lr1to1_score = r2_score(test_U1, pred_A1U1, multioutput='raw_values')
        lr1to2_score = r2_score(test_U2, pred_A1U2, multioutput='raw_values')
        lr2to1_score = r2_score(test_U1, pred_A2U1, multioutput='raw_values')
        lr2to2_score = r2_score(test_U2, pred_A2U2, multioutput='raw_values')

        # train_lr1to1_score = r2_score(train_U1, train_pred_A1U1, multioutput='raw_values')
        # train_lr1to2_score = r2_score(train_U2, train_pred_A1U2, multioutput='raw_values')
        # train_lr2to1_score = r2_score(train_U1, train_pred_A2U1, multioutput='raw_values')
        # train_lr2to2_score = r2_score(train_U2, train_pred_A2U2, multioutput='raw_values')
        #

        # print()
        # print(lr1to2_score)
        # print(lr2to1_score)
        # print(lr1to1_score.mean(), lr2to1_score.mean(), lr2to2_score.mean(), lr1to2_score.mean())
        # print(lr1to1_pearson.mean(), lr2to1_pearson.mean(), lr2to2_pearson.mean(), lr1to2_pearson.mean())
        # print()

        nc_i = U1.shape[1]
        nc_j = U2.shape[1]
        outs[k] = {
            'U1_mean': U1_mean,
            'U1_std': U1_std,
            'U2_mean': U2_mean,
            'U2_std': U2_std,

            'lr1to1': lr1to1,
            'lr1to2': lr1to2,
            'lr2to1': lr2to1,
            'lr2to2': lr2to2,
            'metadata': {'method': method_name, 'num_concepts_i': nc_i, 'num_concepts_j': nc_j},

            # 'train_lr1to1_score': train_lr1to1_score,
            # 'train_lr1to2_score': train_lr1to2_score,
            # 'train_lr2to1_score': train_lr2to1_score,
            # 'train_lr2to2_score': train_lr2to2_score,
            'lr1to1_score': lr1to1_score,
            'lr1to2_score': lr1to2_score,
            'lr2to1_score': lr2to1_score,
            'lr2to2_score': lr2to2_score,

            # 'train_lr1to1_pearson': train_lr1to1_pearson,
            # 'train_lr1to2_pearson': train_lr1to2_pearson,
            # 'train_lr2to1_pearson': train_lr2to1_pearson,
            # 'train_lr2to2_pearson': train_lr2to2_pearson,

            'lr1to1_pearson': lr1to1_pearson,
            'lr1to2_pearson': lr1to2_pearson,
            'lr2to1_pearson': lr2to1_pearson,
            'lr2to2_pearson': lr2to2_pearson,
        }

    if len(outs) == 1:
        return outs[0]
    else:
        return outs



def create_method_folder_name(method, method_dict):

    method_param_name = saving.convert_comparison_method_params(method_dict)
    folder_name = os.path.join(method, method_param_name)
    return folder_name


def compare_model_concepts(methods_list, activations1, activations2, U1, U2):

    comparison_outputs = {}
    for mei, method_dict in enumerate(methods_list):
        method = method_dict['method']
        output_folder = method_dict['method_output_folder']
        if method in ['pearson', 'spearman']:
            out = correlation_comparison(method, U1, U2)
        elif method in ['linear_regression', 'ridge_regression', 'lasso_regression', 'lasso_regression_c', 'elastic_net', 'k_nearest_neighbors', 'radius_neighbors']:
            # method_dict['regression_train_pct'] = train_pct
            # method_dict['seed'] = seed
            if U1 is not None or U2 is not None:
                out = regression_comparison(method, method_dict, activations1, activations2, U1, U2)
            else:
                out = None
        elif method in ['concept_linear_regression', 'concept_ridge_regression', 'concept_lasso_regression', 'concept_lasso_regression_c', 'concept_elastic_net', 'concept_k_nearest_neighbors', 'concept_radius_neighbors']:
            # method_dict['regression_train_pct'] = train_pct
            # method_dict['seed'] = seed
            if U1 is not None and U2 is not None:
                out = concept_regression_comparison(method, method_dict, U1, U2)
            else:
                out = None
        else:
            raise ValueError(f'Unknown method: {method}')
        comparison_outputs[output_folder] = out

    return comparison_outputs


def chunked_compare_model_concepts(compare_args):
    outs = []
    for arg_list in compare_args:
        outs.append(compare_model_concepts(*arg_list))
    return outs


def chunk_list(_list, num_chunks):
    chunk_size = len(_list) // num_chunks
    chunks = []
    for i in range(num_chunks):
        if i == num_chunks - 1:
            chunks.append((_list[i * chunk_size:],))
        else:
            chunks.append((_list[i * chunk_size: (i + 1) * chunk_size],))
    return chunks


def process_config(config, outdir):

    comparison_methods = config['methods']
    train_pct = config.get('regression_train_pct', None)
    num_folds = config.get('num_folds', None)
    patchify = config.get('patchify', False)
    patch_size = config.get('patch_size', None)
    assert (train_pct is None) or (num_folds is None), 'Cannot specify both train_pct and num_folds'
    assert (train_pct is not None) or (num_folds is not None), 'Must specify either train_pct or num_folds'

    seed = config['seed']

    data_group_params = {
        'num_images': config['num_images'],
        'num_image_repeats': config['num_image_repeats'] if not patchify else 1,
        'transform_type': config['transform_type'],
        'seed': seed,
        'regression_train_pct': train_pct,
        'num_folds': num_folds,
        'patchify': patchify,
        'patch_size': patch_size,
    }
    data_group_name = saving.convert_comparison_method_params(data_group_params)

    method_output_folders = []
    for mei, method_dict in enumerate(comparison_methods):
        method = method_dict['method']
        if method in ['pearson', 'spearman']:
            folder_name = create_method_folder_name(method, method_dict)
        elif method in ['linear_regression', 'ridge_regression', 'lasso_regression', 'lasso_regression_c','elastic_net', 'k_nearest_neighbors', 'radius_neighbors']:
            method_dict['regression_train_pct'] = train_pct
            method_dict['num_folds'] = num_folds
            method_dict['seed'] = seed
            folder_name = create_method_folder_name(method, method_dict)
        elif method in ['concept_linear_regression', 'concept_ridge_regression', 'concept_lasso_regression', 'concept_lasso_regression_c', 'concept_elastic_net', 'concept_k_nearest_neighbors', 'concept_radius_neighbors']:
            method_dict['regression_train_pct'] = train_pct
            method_dict['num_folds'] = num_folds
            method_dict['seed'] = seed
            folder_name = create_method_folder_name(method, method_dict)
        elif method in ['clip']:
            folder_name = create_method_folder_name(method, method_dict)
        else:
            raise ValueError(f'Unknown method: {method}')
        m_folder = os.path.join(outdir, data_group_name, folder_name)
        method_output_folders.append(m_folder)
        method_dict['method_output_folder'] = m_folder

    return data_group_name, method_output_folders


def build_model_comparison_parser():
    parser = concept_comparison_parser()
    parser.add_argument('--comparison_config', type=str, required=True)
    parser.add_argument('--folder_exists', type=str, default='skip')
    parser.add_argument('--comparison_output_root', type=str, default='./')
    return parser


def split_image_group(image_list, train_pct):
    num_images = len(image_list)
    num_train = int(train_pct * num_images)
    np.random.shuffle(image_list)
    train_image_list = image_list[:num_train]
    test_image_list = image_list[num_train:]

    return train_image_list, test_image_list


def k_fold_split_image_group(image_list, num_folds, num_image_repeats, patched=False, num_patches=None):

    num_images = len(image_list)
    # create cross-val groups
    indices = np.arange(num_images)
    splits = list(KFold(num_folds).split(indices))

    if not patched:
        repeated_image_list = image_list * num_image_repeats
        train_image_indices = {}
        test_image_indices = {}
        for i in range(num_image_repeats):
            for k, (train_idx, test_idx) in enumerate(splits):
                if k not in train_image_indices:
                    train_image_indices[k] = []
                    test_image_indices[k] = []
                train_image_indices[k].extend(splits[k][0] + i * num_images)
                test_image_indices[k].extend(splits[k][1] + i * num_images)

        # plt.subplots(5, 1, figsize=(20, 5))
        # for i in range(5):
        #     arr = np.zeros(num_images * num_image_repeats)
        #     arr[train_image_indices[i]] = 1
        #     arr[test_image_indices[i]] = 2
        #     plt.subplot(5, 1, i + 1)
        #     plt.scatter(np.arange(num_images * num_image_repeats), arr)
        # plt.show()
    else:
        num_patches = int(num_patches)
        patch_inds = np.arange(num_images * num_patches)
        index_map = {}
        for ind in indices:
            index_map[ind] = patch_inds[ind * num_patches: (ind + 1) * num_patches]

        repeated_image_list = image_list
        train_image_indices = {}
        test_image_indices = {}
        for k, (train_idx, test_idx) in enumerate(splits):
            train_image_indices[k] = np.concatenate([index_map[ind] for ind in splits[k][0]]).tolist()
            test_image_indices[k] = np.concatenate([index_map[ind] for ind in splits[k][1]]).tolist()
        # print()
        # plt.subplots(5, 1, figsize=(20, 5))
        # for i in range(5):
        #     arr = np.zeros(num_images * num_patches)
        #     arr[train_image_indices[i]] = 1
        #     arr[test_image_indices[i]] = 2
        #     plt.subplot(5, 1, i + 1)
        #     plt.scatter(np.arange(num_images * num_patches), arr)
        # plt.show()



    # num_train = int(train_pct * num_images)
    # np.random.shuffle(image_list)
    # train_image_list = image_list[:num_train]
    # test_image_list = image_list[num_train:]

    return repeated_image_list, train_image_indices, test_image_indices


def shared_concept_proposals_inference(params):
    image_group = params['image_group']
    patchify = params['patchify']
    num_images = params['num_images']
    num_image_repeats = params['num_image_repeats']
    num_folds = params['num_folds']
    regression_train_pct = params['regression_train_pct']
    dataset = params['dataset']
    device = params['device']
    class_idx = params['class_idx']
    fe_outs = params['fe_outs']
    transforms = params['transforms']
    act_hooks = params['act_hooks']
    models = params['models']
    patch_size = params.get('patch_size', None)
    comparison_methods = params['comparison_methods']
    target_num_samples = num_images * num_image_repeats
    dataset_name = dataset

    if patchify:

        if dataset == 'imagenet':
            assert transforms[0].transforms[1].size == transforms[1].transforms[1].size
            assert transforms[0].transforms[1].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[1].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[1].size[0] / patch_size)
        elif dataset == 'nabirds':
            assert transforms[0].transforms[0].size == transforms[1].transforms[0].size
            assert transforms[0].transforms[0].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[0].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[0].size[0] / patch_size)
        elif dataset == 'nabirds_modified':
            assert transforms[0].transforms[0].size == transforms[1].transforms[0].size
            assert transforms[0].transforms[0].size[0] == 224
            # These params must be true for this function to work correctly
            num_patches = np.ceil(transforms[0].transforms[0].size[0] / patch_size) * np.ceil(
                transforms[0].transforms[0].size[0] / patch_size)
            dataset_name = 'nabirds'

        image_list, train_indices, test_indices = k_fold_split_image_group(image_group[class_idx], num_folds,
                                                                           1, patched=True,
                                                                           num_patches=num_patches)

    else:
        if len(image_group[class_idx]) < num_images:
            print(f'Scaling number of repeats to match number of available images -- {class_idx}')
            num_image_repeats = target_num_samples // len(image_group[class_idx])

        train_indices = None
        test_indices = None
        # split data into train and test using a percentage
        if regression_train_pct is not None and num_folds is None:
            train_image_list, test_image_list = split_image_group(image_group[class_idx], regression_train_pct)
            # repeat images (for different random crops)
            train_image_list = train_image_list * num_image_repeats
            test_image_list = test_image_list * num_image_repeats
            image_list = train_image_list + test_image_list
            num_train = len(train_image_list)
            num_test = len(test_image_list)
        # split data into equal folds and use each fold as a test set (train multiple regression models)
        else:
            image_list, train_indices, test_indices = k_fold_split_image_group(image_group[class_idx], num_folds,
                                                                               num_image_repeats)

    for mi in range(len(fe_outs)):
        transform = transforms[mi]
        out = ceh.select_class_and_load_images(image_path_list=image_list,
                                               data_root=f'./data/{dataset_name}/',
                                               transform=transform)
        image_size = out['image_size']
        patches = ceh.patchify_images(out['images_preprocessed'], patch_size, strides=None)
        images_preprocessed = patches
        out = _batch_inference(models[mi], images_preprocessed, batch_size=256,  resize=image_size, device=device)
        act_hooks[mi].concatenate_layer_activations()

        print(len(image_list), images_preprocessed.shape)

    # update config with train test information
    for comparison_method in comparison_methods:
        if comparison_method.get('regression_train_pct', None) is not None:
            comparison_method['num_train'] = num_train
            comparison_method['num_test'] = num_test
        elif comparison_method.get('num_folds', None) is not None:
            comparison_method['num_folds'] = {'train_indices': train_indices, 'test_indices': test_indices}


def _process_coeff_for_layer(concept_folder, layer, class_idx, activations):
    concepts = load_concepts(concept_folder, layer, class_idx)
    if concepts is None:
        return None
    W = concepts['W']
    U = compute_concept_coefficients(activations, W, method='fnnls')
    return U


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@torch.no_grad()
def main():
    parser = build_model_comparison_parser()
    parser.add_argument('--multiprocessing', type=int, default=0)
    args = parser.parse_args()

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    dataset = config['dataset']
    num_images = config['num_images']
    num_image_repeats = config.get('num_image_repeats', 1)
    transform_type = config['transform_type']
    seed = config['seed']
    set_seed(seed)
    regression_train_pct = config.get('regression_train_pct', None)
    num_folds = config.get('num_folds', None)
    comparison_methods = config['methods']
    comparison_name = config['comparison_name']
    only_last_layer = config['only_last_layer']
    patchify = config.get('patchify', False)
    patch_size = config.get('patch_size', None)
    move_to_cpu_every = config.get('move_to_cpu_every', None)

    output_dir = build_output_dir(args, comparison_name)
    data_group_name, method_output_folders = process_config(config, output_dir)
    comparison_methods_tmp = []
    for mi, m_folder in enumerate(method_output_folders):
        if os.path.exists(m_folder) and args.folder_exists == 'raise':
            raise ValueError(f'Folder {m_folder} already exists')
        if os.path.exists(m_folder) and args.folder_exists == 'skip':
            print(f'Folder {m_folder} already exists, skipping')
            continue
        os.makedirs(m_folder, exist_ok=True)
        comparison_methods_tmp.append(comparison_methods[mi])
    comparison_methods = comparison_methods_tmp

    print(method_output_folders)

    out = build_model_comparison_param_dicts(args)
    param_dicts1 = out['param_dicts1']
    param_dicts2 = out['param_dicts2']
    concepts_folders = out['concepts_folders']
    igs = args.cross_model_image_group_strategy
    dataset_name = param_dicts1['dataset_params']['dataset_name']

    device = 'cuda'

    fe_outs = []
    # backbones = []
    decomp_method = []
    transforms = []
    act_hooks = []
    models = []
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):
        model_name, ckpt_path = param_dicts['model']
        model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
        model = model_out['model']

        transform = model_out['test_transform'] if transform_type == 'test' or patchify else model_out['transform']
        transforms.append(transform)
        fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
        act_hook = ActivationHook(move_to_cpu_every=move_to_cpu_every)
        act_hook.register_hooks(fe_out['layer_names'], fe_out['layers'], fe_out['post_activation_func'])
        act_hooks.append(act_hook)
        fe_outs.append(fe_out)

        models.append(model)

        # overwrite original num images (need original for accurately loading paths)
        param_dicts['num_images'] = args.cmigs_num_images
        decomp_method.append(param_dicts['dl_params']['decomp_method'])
        param_dicts['dataset_params']['num_images'] = num_images
        param_dicts['dataset_params']['seed'] = seed

    m0_layers = fe_outs[0]['layer_names'][::-1]
    m1_layers = fe_outs[1]['layer_names'][::-1]
    if only_last_layer:
        m0_layers = [m0_layers[0]]
        m1_layers = [m1_layers[0]]

    image_group = create_image_group(strategy=igs, param_dicts=[param_dicts1, param_dicts2])

    class_list = param_dicts1['class_list']
    pbar = tqdm(class_list)
    ci = 0
    for class_idx in pbar:
        ci += 1
        pbar.set_description(f'Class {class_idx}')

        for m_folder in method_output_folders:
            if os.path.exists(m_folder) and args.folder_exists == 'raise':
                raise ValueError(f'Folder {m_folder} already exists')
            os.makedirs(os.path.join(m_folder, f'{class_idx}'), exist_ok=True)

        if class_idx not in image_group or len(image_group[class_idx]) < 5:
            print(f'No images for class {class_idx}')
            continue

        params = dict(
            image_group=image_group, patchify=patchify, num_images=num_images, num_image_repeats=num_image_repeats,
            num_folds=num_folds, regression_train_pct=regression_train_pct, dataset=dataset, device=device,
            class_idx=class_idx, fe_outs=fe_outs, transforms=transforms, act_hooks=act_hooks, models=models,
            comparison_methods=comparison_methods, patch_size=patch_size,
        )
        shared_concept_proposals_inference(params)

        print('Computing concept coefficients')
        st = time.time()
        U1_layers = []
        for li, m0_layer in enumerate(m0_layers):  # reverse order to start from the last layer
            U1 = _process_coeff_for_layer(concepts_folders[0], m0_layer, class_idx, act_hooks[0].layer_activations[m0_layer])

            U1_layers.append(U1)
        print(time.time() - st)

        st = time.time()
        U2_layers = []
        for lj, m1_layer in enumerate(m1_layers):  # reverse order to start from the last layer
            U2 = _process_coeff_for_layer(concepts_folders[1], m1_layer, class_idx, act_hooks[1].layer_activations[m1_layer])
            U2_layers.append(U2)
        print(time.time() - st)

        if args.multiprocessing:
            comparison_args = []
            file_paths = []
            for li, m0_layer in enumerate(m0_layers):  # reverse order to start from the last layer
                activations1 = act_hooks[0].layer_activations[m0_layer]
                for lj, m1_layer in enumerate(m1_layers):  # reverse order to start from the last layer
                    activations2 = act_hooks[1].layer_activations[m1_layer]
                    comparison_args.append(
                        (comparison_methods, activations1, activations2, U1_layers[li], U2_layers[lj]))
                    file_paths.append(os.path.join(f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl'))

            st = time.time()
            with multiprocessing.get_context('spawn').Pool(processes=args.multiprocessing) as pool:
                comparison_outputs_list = pool.starmap(compare_model_concepts, comparison_args)
            print(time.time() - st, 'multiprocessing layerwise comparison')

            # save outputs
            for i, comparison_outputs in enumerate(comparison_outputs_list):
                for output_folder, out in comparison_outputs.items():
                    fp = os.path.join(output_folder, file_paths[i])
                    with open(fp, 'wb') as f:
                        pkl.dump(out, f)
        else:
            print('Comparing model concepts')
            for li, m0_layer in enumerate(m0_layers):  # reverse order to start from the last layer
                activations1 = act_hooks[0].layer_activations[m0_layer]
                for lj, m1_layer in enumerate(m1_layers):  # reverse order to start from the last layer
                    activations2 = act_hooks[1].layer_activations[m1_layer]
                    comparison_outputs = compare_model_concepts(comparison_methods, activations1, activations2,
                                                                U1_layers[li], U2_layers[lj])

                    for output_folder, out in comparison_outputs.items():
                        with open(os.path.join(output_folder, f'{class_idx}', f'{m0_layer}-{m1_layer}.pkl'), 'wb') as f:
                            pkl.dump(out, f)

        for mi in range(len(fe_outs)):
            act_hooks[mi].reset_activation_dict()

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method("spawn")
    main()

'''
1) Convert params to names
- image selection params (num images, transform type, num repeats, dataset)
- method params -> name
    - pearson
    - spearman
    - regression params -> name
        - regression type, penalty amount, standardize, standardize targets, train pct, seed
- add comparison map helper to ceh + saving
    - input should be config -> stringified (through prev steps) -> name
    - name should call back abbr comparison name
    - return comparison name
    - given a comparison name -> should return the full names for all of the parts
    - parts include:
        model1 concept name
        model2 concept name
        comparison image selection name
        method_i name

        

'''
