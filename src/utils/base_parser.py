import argparse
from src.utils import saving
import torch

def build_parser():
    parser = argparse.ArgumentParser()

    # MODEL PARAMS
    parser.add_argument('--model1', type=str, default='resnet18.a2_in1k')
    parser.add_argument('--model1_ckpt', type=str, default=None)
    parser.add_argument('--model2', type=str, default='resnet50.a2_in1k')
    parser.add_argument('--model2_ckpt', type=str, default=None)
    parser.add_argument('--split1', type=str, default='layer_8')
    parser.add_argument('--split2', type=str, default='layer_8')

    # DATASET PARAMS
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--class_list_path', type=str, default=None)

    # CRAFT PARAMS
    parser.add_argument('--num_concepts', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)

    # MODIFIED CRAFT PARAMS
    parser.add_argument('--image_group_strategy', type=str, default='union')
    parser.add_argument('--viz_sim_matrices', action='store_true', default=False)
    parser.add_argument('--viz_crop_concept_scores', action='store_true', default=False)
    parser.add_argument('--skip_viz_crops', action='store_true', default=False)
    parser.add_argument('--save_individual_crops', action='store_true', default=False)
    parser.add_argument('--skip_sensitivity_analysis', action='store_true', default=False)
    parser.add_argument('--m1_seed', type=int, default=0)
    parser.add_argument('--m2_seed', type=int, default=0)

    # CKA PARAMS
    parser.add_argument('--normalize_w', type=bool, default=False)
    parser.add_argument('--normalize_a', type=bool, default=False)
    parser.add_argument('--norm_method', type=str, default=None)
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--no_debiased', dest='debiased', action='store_false')
    parser.add_argument('--skip_cka', action='store_true', default=False)
    parser.add_argument('--use_weighted_concept_vector', action='store_true', default=False)

    # RUN PARAMS
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--force_run_craft', action='store_true', default=False, help='Force run CRAFT')
    parser.add_argument('--start_class_idx', type=int, default=0)
    parser.add_argument('--end_class_idx', type=int, default=1000)

    return parser


def build_param_dicts(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        device = torch.device(f'cuda:{args.gpu_id}')

    model1_tuple = (args.model1, args.model1_ckpt)
    model2_tuple = (args.model2, args.model2_ckpt)

    # parse split_args
    split1 = args.split1.split('_')
    split2 = args.split2.split('_')
    if len(split1) == 1:
        split1.append(None)
    if len(split2) == 1:
        split2.append(None)

    split_params = [{'split_layer': split1[0], 'split_point': int(split1[1])},
                    {'split_layer': split2[0], 'split_point': int(split2[1])}]

    if args.class_list_path is None:
        class_list = range(args.start_class_idx, args.end_class_idx)
    else:
        with open(args.class_list_path, 'r') as f:
            class_list = json.load(f)

    dataset_params = {
        'dataset_name': args.dataset,
        'split': args.dataset_split,
        'num_images': args.num_images,
        'seed': args.dataset_seed,
    }

    sim_params = {
        'kernel': args.kernel,
        'debiased': args.debiased,
        'normalize_w': args.normalize_w,
        'normalize_a': args.normalize_a,
        'norm_method': args.norm_method,
        'use_weighted_concept_vector': args.use_weighted_concept_vector,
    }

    match_params = {'model1_tuple': model1_tuple,
                     'model2_tuple': model2_tuple,
                     'image_group_strategy': args.image_group_strategy,
                     }

    craft_params_1 = {
        'model_name': args.model1,
        'patch_size': args.patch_size,
        'num_concepts': args.num_concepts,
        'viz_sim_matrices': args.viz_sim_matrices,
        'viz_crop_concept_scores': args.viz_crop_concept_scores,
        'skip_viz_crops': args.skip_viz_crops,
        'nmf_seed': args.m1_seed,
    }

    craft_params_2 = {
        'model_name': args.model2,
        'patch_size': args.patch_size,
        'num_concepts': args.num_concepts,
        'viz_sim_matrices': args.viz_sim_matrices,
        'viz_crop_concept_scores': args.viz_crop_concept_scores,
        'skip_viz_crops': args.skip_viz_crops,
        'nmf_seed': args.m2_seed,
    }

    return model1_tuple, model2_tuple, split_params, dataset_params, class_list, craft_params_1, craft_params_2, match_params, sim_params, device


def generate_save_names(split_params, dataset_params, match_params, craft_params, cka_params):
    # GENERATE SAVE NAME
    dataset_name = saving.convert_params(dataset_params)
    craft_name_1 = saving.convert_params(craft_params[0])
    craft_name_2 = saving.convert_params(craft_params[1])
    cka_name = saving.convert_params(cka_params)
    match_name = saving.convert_match_params(match_params)
    sp1_name = saving.convert_split_params(split_params[0])
    sp2_name = saving.convert_split_params(split_params[1])
    return dict(match_name=match_name, sp1_name=sp1_name, sp2_name=sp2_name, craft_name_1=craft_name_1,
                craft_name_2=craft_name_2,
                cka_name=cka_name, dataset_name=dataset_name)