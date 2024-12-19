from sklearn.utils._testing import ignore_warnings
import torch
from math import ceil
from scipy.stats import pearsonr, spearmanr
from scipy.stats._warnings_errors import ConstantInputWarning, NearConstantInputWarning
import numpy as np
import os
import pickle as pkl
from src.dictionary_learning import DictionaryLearner


def load_concepts(concepts_folder, layer, class_idx):
    try:
        print(os.path.join(concepts_folder, layer, f'{class_idx}.pkl'), os.path.exists(os.path.join(concepts_folder, layer, f'{class_idx}.pkl')))
        with open(os.path.join(concepts_folder, layer, f'{class_idx}.pkl'), 'rb') as f:
            concepts = pkl.load(f)
    except FileNotFoundError:
        concepts = None
    return concepts


def compute_concept_coefficients(activations, concepts_w, method, device='cpu', params=None):
    if params is None:
        params = {}
    params['device'] = device
    out = DictionaryLearner.static_transform(method, activations, concepts_w, params=params)
    # print(out['err'])
    return out['U']


@ignore_warnings(category=UserWarning)
def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
    '''
    Code from CRAFT repository
    '''
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i * batch_size for i in range(nb_batchs)]

    results = []

    with torch.no_grad():
        for i in start_ids:
            x = torch.tensor(dataset[i:i + batch_size])
            x = x.to(device)

            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results

@ignore_warnings(category=ConstantInputWarning)
def correlation_comparison(method, Ui, Uj, self_compare=False):
    """
    Compare two coefficient matrices Ui and Uj
    :param Ui: torch.Tensor
    :param Uj: torch.Tensor
    :return: List of dictionaries with each comparison method
    """
    if Ui is None or Uj is None:
        return None
    num_concepts_i = Ui.shape[1]
    num_concepts_j = Uj.shape[1]
    statistic_dict = {}
    statistic_dict1to2 = {}

    for i in range(num_concepts_i):
        for j in range(num_concepts_j):
            if method == 'pearson':
                out1to2 = pearsonr(Ui[:, i], Uj[:, j])
            elif method == 'spearman':
                out1to2 = spearmanr(Ui[:, i], Uj[:, j])
            else:
                raise ValueError(f'Unknown method: {method}')
            statistic_dict1to2[(i, j)] = out1to2

    statistic_dict['metadata'] = {'method': method, 'num_concepts_i': num_concepts_i,
                                  'num_concepts_j': num_concepts_j}
    statistic_dict['1to2'] = statistic_dict1to2

    if self_compare:
        statistic_dict1to1 = {}
        statistic_dict2to2 = {}
        for i in range(num_concepts_i):
            for j in range(num_concepts_j):
                if method == 'pearson':
                    out1to1 = pearsonr(Ui[:, i], Ui[:, j])
                elif method == 'spearman':
                    out1to1 = spearmanr(Ui[:, i], Ui[:, j])
                else:
                    raise ValueError(f'Unknown method: {method}')
                statistic_dict1to1[(i, j)] = out1to1

        for i in range(num_concepts_i):
            for j in range(num_concepts_j):
                if method == 'pearson':
                    out2to2 = pearsonr(Uj[:, i], Uj[:, j])
                elif method == 'spearman':
                    out2to2 = spearmanr(Uj[:, i], Uj[:, j])
                else:
                    raise ValueError(f'Unknown method: {method}')
                statistic_dict2to2[(i, j)] = out2to2

        statistic_dict['1to1'] = statistic_dict1to1
        statistic_dict['2to2'] = statistic_dict2to2

    return statistic_dict


def convert_to_correlation_comparison_to_array(statistic_dict, metadata):

    arr = np.zeros((metadata['num_concepts_i'], metadata['num_concepts_j']))
    for i in range(metadata['num_concepts_i']):
        for j in range(metadata['num_concepts_j']):
            key = (i, j)
            value = statistic_dict[key]
            arr[key[0], key[1]] = value.statistic

    return arr