import torch
from math import ceil
from src.utils.parser_helper import concept_extraction_parser
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHook
from src import eval_model
import json
import os
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
import numpy as np


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


def join_image_groups(strategy, image_group1, image_group2):
    image_group = {}
    keys = image_group1.keys()
    for key in keys:
        if strategy == 'union':
            image_group[key] = list(set(image_group1.get(key, []) + image_group2.get(key, [])))
        elif strategy == 'intersection':
            image_group[key] = list(set(image_group1[key]).intersection(image_group2[key]))
        elif strategy == 'model1':
            image_group[key] = image_group1[key]
        elif strategy == 'model2':
            image_group[key] = image_group2[key]
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
    return image_group


def create_image_group(strategy, param_dicts, return_eval_dict=False):
    if strategy == 'craft':
        # Load/compute model predictions
        dataset_params = param_dicts['dataset_params']
        dataset = dataset_params['dataset_name']
        dataset_seed = dataset_params['seed']
        split = dataset_params['split']
        model_name, ckpt_path = param_dicts['model']
        model_eval_path = f'model_evaluation/{dataset}/{model_name}_{split}.json'
        try:
            with open(model_eval_path, 'r') as f:
                eval_dict = json.load(f)
                predictions = eval_dict['predictions']
        except FileNotFoundError:
            predictions = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data')['predictions']

        # load images
        print('Loading images')
        image_group = ceh.group_images(method='craft', params={'predictions': predictions,
                                                               'num_images': dataset_params['num_images'],
                                                               'seed': dataset_seed})
    elif strategy == 'union-craft':  # requires two models
        image_groups = []
        eval_dict = []
        for pd in param_dicts:
            # Load/compute model predictions
            dataset_params = pd['dataset_params']
            dataset = dataset_params['dataset_name']
            dataset_seed = dataset_params['seed']
            split = dataset_params['split']
            model_name, ckpt_path = pd['model']
            model_eval_path = f'model_evaluation/{dataset}/{model_name}_{split}.json'
            try:
                with open(model_eval_path, 'r') as f:
                    _eval_dict = json.load(f)
                    predictions = _eval_dict['predictions']
                    eval_dict.append(_eval_dict)
            except FileNotFoundError:
                _eval_dict = eval_model.main(model_name, dataset, split, ckpt_path, data_root='./data')
                predictions = _eval_dict['predictions']
                eval_dict.append(_eval_dict)

            # load images
            print('Loading images')
            image_group = ceh.group_images(method='craft', params={'predictions': predictions,
                                                                   'num_images': None,
                                                                   'seed': dataset_seed})
            image_groups.append(image_group)

        image_group = join_image_groups('union', image_groups[0], image_groups[1])

        # subsample N images per class
        assert param_dicts[0]['dataset_params']['num_images'] == param_dicts[1]['dataset_params']['num_images']
        assert param_dicts[0]['dataset_params']['seed'] == param_dicts[1]['dataset_params']['seed']
        target_num_images = param_dicts[0]['dataset_params']['num_images']
        seed = param_dicts[0]['dataset_params']['seed']
        if target_num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in image_group.keys():
                path_list = sorted(image_group[i])
                if target_num_images > len(path_list):
                    print(f'Warning: class {i} has only {len(path_list)} / {target_num_images} images')
                    num_images = len(path_list)
                else:
                    num_images = target_num_images

                subsampled_label_groups[i] = list(rng.choice(path_list, size=num_images, replace=False))

            image_group = subsampled_label_groups

    else:
        raise ValueError(f'Unknown image_group_strategy: {strategy}')

    if return_eval_dict:
        return image_group, eval_dict
    return image_group


def transform_images(image_path_list, dataset_name, model_out, param_dicts, transform_type='patchify'):
    if dataset_name == 'nabirds_modified' or dataset_name == 'nabirds_stanford_cars':
        dataset_name = 'nabirds'
    if transform_type == 'patchify':
        out = ceh.select_class_and_load_images_v2(image_path_list=image_path_list,
                                                  data_root=f'./data/{dataset_name}/',
                                                  transform=model_out['test_transform'])
        if out is None:
            print(f'Class {class_idx} not found in image group')
            return None

        image_size = out['image_size']
        patch_size = param_dicts['feature_extraction_params']['patch_size']
        images = ceh.patchify_images(out['images_preprocessed'], patch_size, strides=None)
    elif transform_type == 'test':
        num_image_repeats = param_dicts['feature_extraction_params']['num_image_repeats']
        out = ceh.select_class_and_load_images_v3(image_path_list=image_path_list,
                                                  data_root=f'./data/{dataset_name}/',
                                                  transform=model_out['test_transform'],
                                                  num_repeats=num_image_repeats)

        if out is None:
            print(f'Class {class_idx} not found in image group')
            return None

        image_size = out['image_size']
        images = out['images_preprocessed']
    elif transform_type == 'train':
        num_image_repeats = param_dicts['feature_extraction_params']['num_image_repeats']
        out = ceh.select_class_and_load_images_v3(image_path_list=image_path_list,
                                                  data_root=f'./data/{dataset_name}/',
                                                  transform=model_out['transform'],
                                                  num_repeats=num_image_repeats)
        if out is None:
            print(f'Class {class_idx} not found in image group')
            return None

        image_size = out['image_size']
        images = out['images_preprocessed']
    else:
        raise ValueError(f'Unknown transform type: {transform_type}')

    return dict(images=images, image_size=image_size)


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


if __name__ == '__main__':
    parser = concept_extraction_parser()
    args = parser.parse_args()
    param_dicts, save_names = build_param_dicts(args)
    force_run = True

    activations_dir = save_names['activations_dir']
    visualization_dir = save_names['visualization_dir']
    if not os.path.exists(activations_dir) or force_run:
        os.makedirs(activations_dir, exist_ok=True)
    else:
        raise ValueError('Output directory already exists')

    if not os.path.exists(visualization_dir) or force_run:
        os.makedirs(visualization_dir, exist_ok=True)
    else:
        raise ValueError('Visualization directory already exists')

    # Load model
    model_name, ckpt_path = param_dicts['model']
    model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
    model = model_out['model']

    # Insert hooks to track activations
    fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])
    act_hook = ActivationHook(move_to_cpu_in_hook=args.move_to_cpu_in_hook, move_to_cpu_every=args.move_to_cpu_every)
    act_hook.register_hooks(fe_out['layer_names'], fe_out['layers'], fe_out['post_activation_func'])

    # Group images according to a strategy
    igs = param_dicts['feature_extraction_params']['image_group_strategy']
    image_group = create_image_group(strategy=igs, param_dicts=param_dicts)

    class_list = param_dicts['class_list']
    activations_folder = os.path.join(activations_dir, 'activations')
    for layer in fe_out['layer_names']:
        os.makedirs(os.path.join(activations_folder, layer), exist_ok=True)
    dataset_name = param_dicts['dataset_params']['dataset_name']

    print('Extracting activations...')
    pbar = tqdm(class_list)
    for class_idx in pbar:
        if class_idx not in image_group.keys():
            print(f'Class {class_idx} not found in image group')
            continue
        transform_out = transform_images(image_group[class_idx], dataset_name, model_out, param_dicts,
                                         transform_type=args.transform)
        images = transform_out['images']
        image_size = transform_out['image_size']

        # Extract model activations
        preds = _batch_inference(model, images, batch_size=args.batch_size,
                                 resize=image_size,
                                 device=param_dicts['device'])
        act_hook.concatenate_layer_activations()

        for layer in act_hook.layer_activations.keys():
            torch.save(act_hook.layer_activations[layer], os.path.join(activations_folder, layer, f'{class_idx}.pth'))

        act_hook.reset_activation_dict()
