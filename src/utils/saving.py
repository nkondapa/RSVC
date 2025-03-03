import os

model_names = {
    'vit_small_patch16_224.augreg_in21k_ft_in1k': 'vit_s',
    'vit_base_patch16_224.augreg_in21k_ft_in1k': 'vit_b',
    'vit_large_patch16_224.augreg_in21k_ft_in1k': 'vit_l',
    'resnet18.a2_in1k': 'r18',
    'resnet34.a2_in1k': 'r34',
    'resnet50.a2_in1k': 'r50',
    'nabirds_dino_a3_seed=4834586': 'nb_din_b_s483',
    'nabirds_mae_a3_seed=4834586': 'nb_mae_b_s483',
    'nabirds_dino_a3_seed=87363356': 'nb_din_b_s873',
    'nabirds_mae_a3_seed=87363356': 'nb_mae_b_s873',
    'nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586': 'nb_syn_m0=0.7_t0_r18_s483',
    'nabirds_mod0=0.0_test0_r18_fs_a3_seed=4834586': 'nb_syn_m0=0.0_t0_r18_s483',
    'nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586': 'nb_syn_mall=0.5_t0_r18_s483',
    'nabirds_exc_rwb_r18_fs_a3_seed=4834586_retrained_head': 'nb_exc_rwb_rth_r18_s483',
    'nabirds_exc_rwbamv0_r18_fs_a3_seed=4834586_retrained_head': 'nb_exc_rwbamv0_rth_r18_s483',
    'nabirds_r18_fs_bl_a3_seed=4834586': 'nb_fs_baseline_r18_s483',
    'nabirds_r18_fs_a3_seed=4834586': 'nb_fs_r18_s483',
    'nabirds_r18_fs_a3_seed=87363356': 'nb_fs_r18_s873',
    'nabirds_r18_pt_a3_seed=4834586': 'nb_pt_r18_s483',
    'nabirds_exc_bop_v0_r18_fs_seed=4834586_rh': 'nb_exc_bop_v0_r18_fs_s483_rh',
    'nabirds_exc_wb_v0_r18_fs_seed=4834586_rh': 'nb_exc_wb_v0_r18_fs_s483_rh',
    'nabirds_exc_wb2_v0_r18_fs_seed=4834586_rh': 'nb_exc_wb2_v0_r18_fs_s483_rh',
    'nabirds_exc_bop_v0_r18_pt_seed=4834586_rh': 'nb_exc_bop_v0_r18_pt_s483_rh',
    'nabirds_exc_wb_v0_r18_pt_seed=4834586_rh': 'nb_exc_wb_v0_r18_pt_s483_rh',
    'nabirds_stanford_cars_r18_fs_seed=4834586_rh_nbsc': 'nbsc_r18_fs_s483_rh_nbsc',
    'nabirds_r18_fs_a3_seed=4834586_rh_nbsc': 'nb_r18_fs_s483_rh_nbsc',
}

param_value_names = {
    'image_group_strategy': {'union': 'u', 'intersection': 'i', 'craft': 'c'},
    'cross_model_image_group_strategy': {'union-craft': 'uc'},
    'normalize_w': {True: 'nw', False: None},
    'normalize_a': {True: 'na', False: None},
    'norm_method': {None: None, 'min_max': 'mm', 'z_score': 'zs'},
    "patch_size": lambda x: f'{x}' if x is not None else None,
    'num_concepts': lambda x: f'{x}',
    'seed': lambda x: f'{x}',
    'split_params': lambda x: convert_split_params(x),
    'match_name': lambda x: convert_model_tuple(x[0]) + '_v_' + convert_model_tuple(x[1]),
    'match_params': lambda x: convert_match_params(x),
    'kernel': {'linear': 'lin', 'rbf': 'rbf'},
    'debiased': {True: 'deb', False: None},
    'viz_sim_matrices': lambda x: '',
    'viz_crop_concept_scores': lambda x: '',
    'skip_viz_crops': lambda x: '',
    'save_individual_crops': lambda x: '',
    'skip_cka': lambda x: '',
    'dataset_name': {'imagenet': 'in', 'nabirds': 'nb', 'cub': 'cub', 'nabirds_modified': 'nb_mod', 'nabirds_stanford_cars': 'nbsc'},
    'split': {'train': 'tr', 'val': 'val', 'test': 'test'},
    'num_images': lambda x: f'{x}',
    'crop_size': lambda x: f'{x}',
    'left_to_right': {True: 'ltr', False: 'rtl'},
    'concept_crop_sampling': {'default': '', 'one_per_image': 'opi'},
    'use_weighted_concept_vector': {True: 'wcv', False: ''},
    'nmf_seed': lambda x: f'{x}',
    'model_name': lambda x: f'{x}',
    'data_source': {'crops': 'crops', 'train': 'tr', 'val': 'val', 'test': 'test'},
    'steps': lambda x: f'{x}',
    'decomp_method': {'nmf': 'nmf', 'cnmf': 'cnmf', 'pca': 'pca', 'snmf': 'snmf'},
    'feature_layer_version': lambda x: f'{x}',
    'transform': {'patchify': '', 'test': 'tr=te', 'train': 'tr=tr'},
    "num_image_repeats": lambda x: f'{x}' if x is not None else None,
}
param_key_names = {
    'image_group_strategy': 'igs=',
    'cross_model_image_group_strategy': 'cmigs=',
    'normalize_w': '',
    'normalize_a': '',
    'norm_method': 'nm=',
    'patch_size': 'ps=',
    'num_concepts': 'nc=',
    'split_params': '',
    'match_params': '',
    'match_name': '',
    'seed': 'seed=',
    'kernel': 'kern=',
    'debiased': '',
    'viz_sim_matrices': '',
    'viz_crop_concept_scores': '',
    'skip_viz_crops': '',
    'save_individual_crops': '',
    'skip_cka': '',
    'dataset_name': 'dn=',
    'split': 'spl=',
    'num_images': 'ni=',

    # concept add params
    'crop_size': 'cs=',
    'left_to_right': 'dir=',

    # concept crop sampling
    'concept_crop_sampling': '',

    'use_weighted_concept_vector': '',
    'nmf_seed': 'nmfsd=',
    'model_name': '',

    'data_source': 'ds=',
    'steps': 'steps=',
    'decomp_method': 'dm=',
    'feature_layer_version': 'flv=',
    'transform': '',
     "num_image_repeats": 'nir=',
}


comparison_method_key_names = {
    'alpha': 'a=',
    'standardize': '',
    'standardize_targets': '',
    "num_images": 'ni=',
    "num_image_repeats": 'nir=',
    "transform_type": 'tt=',
    "seed": 'seed=',
    "regression_train_pct": "trpct=",
    "num_folds": "nf=",
    "n_neighbors": "nn=",
    "method": "",
    "regression_params": "",
    "patchify": '',
    "patch_size": "ps=",
    "topk": "topk="
}
comparison_method_param_value_names = {
    'alpha': lambda x: f'{x}',
    'standardize': {True: 'std_in', False: ''},
    'standardize_targets': {True: 'std_tg', False: ''},
    "num_images": lambda x: f'{x}',
    "num_image_repeats": lambda x: f'{x}' if x is not None else None,
    "transform_type": {'train': 'tr', 'val': 'val', 'test': 'te'},
    "seed": lambda x: f'{x}',
    "regression_train_pct": lambda x: f'{x}',
    "num_folds": lambda x: f'{x}' if x is not None else None,
    "n_neighbors": lambda x: f'{x}',
    "method": lambda x: "",
    "regression_params": lambda x: convert_comparison_method_params(x),
    "patchify": {True: 'ptc=t', False: ''},
    "patch_size": lambda x: f'{x}' if x is not None else None,
    "topk": lambda x: f'{x}'
}


def convert_comparison_method_params(params):
    name = ''
    for key, val in params.items():
        if key in comparison_method_key_names:
            kn = comparison_method_key_names[key]
            if callable(comparison_method_param_value_names[key]):
                v = comparison_method_param_value_names[key](val)
            else:
                v = comparison_method_param_value_names[key][val]
            if v:
                name = name + kn + v + '_'
        elif key == 'backbone' or key == 'transform':
            continue
        else:
            raise ValueError(f'Unknown key {key}')

    return name[:-1]


def convert_match_params(match_params):
    model1, model2 = match_params['model1_tuple'], match_params['model2_tuple']
    del match_params['model1_tuple']
    del match_params['model2_tuple']
    igs = match_params['image_group_strategy']

    m1 = convert_model_tuple(model1)# + '_' + convert_split_params(model1['split_params'])
    m2 = convert_model_tuple(model2)# + '_' + convert_split_params(model2['split_params'])

    name = convert_params(match_params)
    return f'{m1}_v_{m2}_{name}'

def convert_model_tuple(model_tuple):
    model_name, ckpt_path = model_tuple
    model_abbr = model_names[model_name]
    if ckpt_path is None:
        ckpt_name = None
    else:
        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]

    name = f'{model_abbr}_ckpt={ckpt_name}'
    return name


def convert_split_params(split_params):
    split_layer = split_params['split_layer']
    split_point = split_params.get('split_point', None)
    if split_point is not None:
        split_point = f'{split_point}'
    else:
        split_point = ''
    name = f'{split_layer}_{split_point}'
    return name


def convert_params(run_params):

    name = ''
    for key, val in run_params.items():
        if key in param_value_names:
            kn = param_key_names[key]
            if callable(param_value_names[key]):
                v = param_value_names[key](val)
            else:
                v = param_value_names[key][val]
            if v:
                name = name + kn + v + '_'
        else:
            raise ValueError(f'Unknown key {key}')

    return name[:-1]


def build_output_dir(output_root, folder_name, comparison_name):
    path = os.path.join(output_root, f'outputs/data/{folder_name}', comparison_name)
    os.makedirs(path, exist_ok=True)
    return path