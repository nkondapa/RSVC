import argparse
from argparse import Namespace
from src.utils import saving
import torch
import json
import os


def concept_extraction_parser():
    parser = argparse.ArgumentParser()

    # MODEL PARAMS
    parser.add_argument('--model', type=str, default='resnet18.a2_in1k')
    parser.add_argument('--model_ckpt', type=str, default=None)

    # DATASET PARAMS
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--dataset_seed', type=int, default=0)

    # DICTIONARY LEARNING PARAMS
    parser.add_argument('--decomp_method', type=str, default='nmf')
    parser.add_argument('--num_concepts', type=int, default=10)
    parser.add_argument('--init_seed', type=int, default=0)

    # FEATURE EXTRACTION PARAMS
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--feature_layer_version', type=str, default="v1")
    parser.add_argument('--image_group_strategy', type=str, default='craft')
    parser.add_argument('--transform', type=str, default='patchify', help='patchify (test + patchify), test (no patchify), or train')
    parser.add_argument('--num_image_repeats', type=int, default=1)
    parser.add_argument('--only_last_layer', action='store_true')

    # RUN PARAMS
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--start_class_idx', type=int, default=0)
    parser.add_argument('--end_class_idx', type=int, default=1000)
    parser.add_argument('--class_list_path', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='./')
    parser.add_argument('--move_to_cpu_every', type=int, default=None)
    parser.add_argument('--move_to_cpu_in_hook', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)

    return parser


def build_param_dicts(args, force_run=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        device = torch.device(f'cuda:{args.gpu_id}')

    model_tuple = (args.model, args.model_ckpt)

    fe_layer_version = args.feature_layer_version
    class_list_values = None
    if args.class_list_path is None:
        class_list = range(args.start_class_idx, args.end_class_idx)
        class_list_values = [0] * len(class_list)
    else:
        with open(args.class_list_path, 'r') as f:
            class_list = json.load(f)
            if type(class_list) is dict:
                class_list_values = list(class_list.values())
                class_list = list(map(int, list(class_list.keys())))
            else:
                class_list_values = [1] * len(class_list)

    dataset_params = {
        'dataset_name': args.dataset,
        'split': args.dataset_split,
        'num_images': args.num_images,
        'seed': args.dataset_seed,
    }

    dl_params = {
        'decomp_method': args.decomp_method,
        'num_concepts': args.num_concepts,
        'seed': args.init_seed,
    }

    feature_extraction_params = {
        'patch_size': args.patch_size if args.transform == 'patchify' else None,
        'feature_layer_version': fe_layer_version,
        'image_group_strategy': args.image_group_strategy,
        'transform': args.transform,
        'num_image_repeats': args.num_image_repeats if args.transform != 'patchify' else None,
    }

    out = {
        'model': model_tuple,
        'dataset_params': dataset_params,
        'class_list': class_list,
        'class_list_values': class_list_values,
        'dl_params': dl_params,
        'feature_extraction_params': feature_extraction_params,
        'device': device,
    }

    model_name = saving.convert_model_tuple(model_tuple)
    dataset_name = saving.convert_params(dataset_params)
    dl_name = saving.convert_params(dl_params)
    fe_name = saving.convert_params(feature_extraction_params)

    activations_dir = os.path.join(args.output_root, 'outputs', 'data', dataset_name, model_name, fe_name)
    concepts_dir = os.path.join(args.output_root, 'outputs', 'data', dataset_name, model_name, fe_name, dl_name)
    visualization_dir = os.path.join(args.output_root, 'outputs', 'visualizations', dataset_name, model_name, fe_name,
                                     dl_name)

    print('Output directory:', activations_dir)

    save_names = {
        'activations_dir': activations_dir,
        'visualization_dir': visualization_dir,
        'concepts_dir': concepts_dir,
        'model_name': model_name,
        'dataset_params': dataset_name,
        'dl_params': dl_name,
        'feature_extraction_params': fe_name,
    }

    return out, save_names


def concept_comparison_parser():
    parser = argparse.ArgumentParser()

    default_model_names = ['resnet18.a2_in1k', 'resnet50.a2_in1k']
    for i in range(2):
        # MODEL PARAMS
        parser.add_argument(f'--model_{i}', type=str, default=default_model_names[i])
        parser.add_argument(f'--model_ckpt_{i}', type=str, default=None)

        # DATASET PARAMS
        parser.add_argument(f'--dataset_{i}', type=str, default='imagenet')
        parser.add_argument(f'--dataset_split_{i}', type=str, default='train')
        parser.add_argument(f'--num_images_{i}', type=int, default=100)
        parser.add_argument(f'--dataset_seed_{i}', type=int, default=0)

        # DICTIONARY LEARNING PARAMS
        parser.add_argument(f'--decomp_method_{i}', type=str, default='nmf')
        parser.add_argument(f'--num_concepts_{i}', type=int, default=10)
        parser.add_argument(f'--init_seed_{i}', type=int, default=0)

        # FEATURE EXTRACTION PARAMS
        parser.add_argument(f'--patch_size_{i}', type=int, default=64)
        parser.add_argument(f'--feature_layer_version_{i}', type=str, default="v1")
        parser.add_argument(f'--image_group_strategy_{i}', type=str, default='craft')
        parser.add_argument(f'--concept_root_folder_{i}', type=str, default='./')
        parser.add_argument(f'--transform_{i}', type=str, default='patchify',
                            help='patchify (test + patchify), test (no patchify), or train')
        parser.add_argument(f'--num_image_repeats_{i}', type=int, default=1)

    parser.add_argument(f'--cross_model_image_group_strategy', type=str, default='union-craft')
    parser.add_argument(f'--cmigs_num_images', type=int, default=100, help='num images per model to compute coeffs')

    # RUN PARAMS
    parser.add_argument('--output_root', type=str, default='./')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--class_list_path', type=str, default=None)
    parser.add_argument('--start_class_idx', type=int, default=0)
    parser.add_argument('--end_class_idx', type=int, default=1000)


    return parser


def split_args(args):
    args1 = {}
    args2 = {}

    for key in vars(args):
        if key.endswith('_0'):
            args1[key[:-2]] = getattr(args, key)
        elif key.endswith('_1'):
            args2[key[:-2]] = getattr(args, key)
        else:
            args1[key] = getattr(args, key)
            args2[key] = getattr(args, key)

    return Namespace(**args1), Namespace(**args2)


def build_model_comparison_param_dicts(args):
    args1, args2 = split_args(args)
    param_dicts1, save_names1 = build_param_dicts(args1)
    param_dicts2, save_names2 = build_param_dicts(args2)

    if save_names1['concepts_dir'][:2] == './':
        save_names1['concepts_dir'] = os.path.join(args1.concept_root_folder, save_names1['concepts_dir'].lstrip('./'))
    if save_names2['concepts_dir'][:2] == './':
        save_names2['concepts_dir'] = os.path.join(args2.concept_root_folder, save_names2['concepts_dir'].lstrip('./'))

    igs = args.cross_model_image_group_strategy

    concepts_folder1 = os.path.join(save_names1['concepts_dir'], 'concepts')
    concepts_folder2 = os.path.join(save_names2['concepts_dir'], 'concepts')
    concepts_folders = [concepts_folder1, concepts_folder2]

    out = {
        'param_dicts1': param_dicts1,
        'save_names1': save_names1,
        'param_dicts2': param_dicts2,
        'save_names2': save_names2,
        'concepts_folders': concepts_folders,
    }

    return out