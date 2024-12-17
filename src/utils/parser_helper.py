import argparse


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