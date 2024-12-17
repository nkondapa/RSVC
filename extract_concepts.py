import torch
from math import ceil
from src.utils.parser_helper import concept_extraction_parser
from src.utils import saving, model_loader, concept_extraction_helper as ceh
import json
import os
from tqdm import tqdm
from src.dictionary_learning import DictionaryLearner
from extract_model_activations import build_param_dicts
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def load_activations(activations_root_folder, layer, class_idx):
    try:
        activations = torch.load(os.path.join(activations_root_folder, layer, f'{class_idx}.pth'))
        return activations
    except FileNotFoundError:
        return None


@ignore_warnings(category=ConvergenceWarning)
def main():
    parser = concept_extraction_parser()
    args = parser.parse_args()
    param_dicts, save_names = build_param_dicts(args, force_run=True)

    # Load model
    model_name, ckpt_path = param_dicts['model']
    model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
    model = model_out['model']

    # Insert hooks to track activations
    fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])

    class_list = param_dicts['class_list']
    activations_folder = os.path.join(save_names['activations_dir'], 'activations')
    concepts_folder = os.path.join(save_names['concepts_dir'], 'concepts')
    os.makedirs(concepts_folder, exist_ok=True)
    dataset_name = param_dicts['dataset_params']['dataset_name']

    num_layers = len(fe_out['layer_names'])
    for li, layer in enumerate(fe_out['layer_names'][::-1]):  # reverse order to start from the last layer
        print(f'Extracting concepts for layer {layer} : {li+1}/{num_layers}')
        layer_folder = os.path.join(concepts_folder, layer)
        os.makedirs(layer_folder, exist_ok=True)

        pbar = tqdm(class_list)
        for class_idx in pbar:
            if os.path.exists(os.path.join(layer_folder, f'{class_idx}.pkl')):
                continue
            pbar.set_description(f'Class {class_idx}')
            activations = load_activations(activations_folder, layer, class_idx)
            if activations is None:
                print(f'Activations for class {class_idx} not found. Skipping...')
                continue
            # Generate Concepts
            dl_params = param_dicts['dl_params']
            dl = DictionaryLearner(dl_params['decomp_method'], params=dl_params)
            out = dl.fit_transform(activations)

            # Save concepts
            dl.save(out, path=os.path.join(layer_folder, f'{class_idx}.pkl'))

        if args.only_last_layer:
            break

if __name__ == '__main__':
    main()