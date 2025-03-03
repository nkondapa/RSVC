import torch
from src.utils.parser_helper import concept_comparison_parser
from src.utils import saving, model_loader, concept_extraction_helper as ceh
import json
import os
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from extract_model_activations import create_image_group
import pickle as pkl
from src.utils.parser_helper import build_model_comparison_param_dicts
from src.utils.funcs import _batch_inference, correlation_comparison, load_concepts, compute_concept_coefficients
from src.utils.model_loader import split_model
from compare_models import process_config



def concept_integrated_gradients(u, w, head, class_idx, steps=30, device='cuda', activations=None):
    '''

    :param u: the concept coefficients
    :param w: concept weight vectors
    :param head: the head of the model
    :param steps: number of steps for the integrated gradients
    :return:
    '''
    x = u
    baseline = torch.zeros_like(x).to(device)[:, None, :]

    B = x.shape[0]

    alpha = torch.linspace(0.0, 1.0, steps).to(device)
    int_x = x[:, None, :]
    int_x = int_x.repeat(1, steps, 1)

    int_x = baseline + alpha[None, :, None] * (int_x - baseline)

    int_x.requires_grad_(True)
    recon_act = torch.einsum('bsc,cd->bsd', int_x, w)
    target = torch.ones((B * steps), dtype=torch.long).to(device) * class_idx

    pred = head(recon_act.reshape(B * steps, -1))

    loss_val = torch.nn.functional.cross_entropy(pred, target, reduction='sum')
    loss_val.backward()

    gradients = int_x.grad
    trapezoidal_gradients = gradients[:, :-1] + gradients[:, 1:]
    averaged_gradients = trapezoidal_gradients.mean(dim=1) * 0.5

    integrated_gradients = (x - baseline.squeeze(1)) * averaged_gradients

    return integrated_gradients


def load_activations(activations_root_folder, layer, class_idx):
    activations = torch.load(os.path.join(activations_root_folder, layer, f'{class_idx}.pth'))
    return activations


def load_concepts(concepts_folder, layer, class_idx):
    try:
        with open(os.path.join(concepts_folder, layer, f'{class_idx}.pkl'), 'rb') as f:
            concepts = pkl.load(f)
    except FileNotFoundError:
        concepts = None
    return concepts


# def build_output_dir(args):
#     fname = 'integrated_gradients'
#     if hasattr(args, 'patchify') and args.patchify:
#         fname += '_patched'
#         print(fname)
#     path = os.path.join(args.importance_output_root, 'outputs/data/concept_importance', args.comparison_save_name, fname)
#     os.makedirs(path, exist_ok=True)
#     return path

def build_output_dir(args, config, save_names1, save_names2, data_group_name=None):
    comparison_name = os.path.join(config['comparison_name'], 'integrated_gradients')
    output_dir = saving.build_output_dir(args.importance_output_root, 'concept_importance', comparison_name)
    if data_group_name is None:
        # process config will break if run twice, pass data_group_name directly if it is already known
        data_group_name, method_output_folders = process_config(config, output_dir)
    model1_out_dir = os.path.join(output_dir, data_group_name, save_names1['model_name'])
    model2_out_dir = os.path.join(output_dir, data_group_name, save_names2['model_name'])
    os.makedirs(model1_out_dir, exist_ok=True)
    os.makedirs(model2_out_dir, exist_ok=True)
    return model1_out_dir, model2_out_dir


@ignore_warnings(category=ConvergenceWarning)
def main():
    parser = concept_comparison_parser()
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--importance_output_root', type=str, default='./')
    parser.add_argument('--patchify', action='store_true')
    parser.add_argument('--comparison_config', type=str, default=None)

    args = parser.parse_args()

    with open(args.comparison_config, 'r') as f:
        config = json.load(f)

    out = build_model_comparison_param_dicts(args)
    param_dicts1 = out['param_dicts1']
    save_names1 = out['save_names1']
    param_dicts2 = out['param_dicts2']
    save_names2 = out['save_names2']
    concepts_folders = out['concepts_folders']
    igs = args.cross_model_image_group_strategy
    dataset_name = param_dicts1['dataset_params']['dataset_name']

    # output_dir = build_output_dir(args)
    model1_out_dir, model2_out_dir = build_output_dir(args, config, save_names1, save_names2)
    output_dirs = [model1_out_dir, model2_out_dir]

    fe_outs = []
    models = []
    backbones = []
    clfs = []
    decomp_methods = []
    for mi, param_dicts in enumerate([param_dicts1, param_dicts2]):

        model_name, ckpt_path = param_dicts['model']
        model_out = model_loader.load_model(model_name, ckpt_path, device=param_dicts['device'], eval=True)
        model = model_out['model']
        backbone, fc = split_model(model)

        fe_out = ceh.load_feature_extraction_layers(model, param_dicts['feature_extraction_params'])

        fe_outs.append(fe_out)
        models.append(model)

        # overwrite original num images (need original for accurately loading paths)
        param_dicts['dataset_params']['num_images'] = args.cmigs_num_images

        backbones.append(backbone)
        clfs.append(fc)
        decomp_methods.append(param_dicts['dl_params']['decomp_method'])

    image_group = create_image_group(strategy=igs, param_dicts=[param_dicts1, param_dicts2])

    if dataset_name == 'nabirds_modified' or dataset_name == 'nabirds_stanford_cars':
        data_root = f'./data/nabirds/'
    else:
        data_root = f'./data/{dataset_name}/'
        
    class_list = param_dicts1['class_list']
    pbar = tqdm(class_list)
    for class_idx in pbar:
        pbar.set_description(f'Class {class_idx}')

        if not args.patchify:
            if class_idx not in image_group:
                print(f'Class {class_idx} not in image group. Skipping...')
                continue
            image_list = image_group[class_idx]
            out = ceh.select_class_and_load_images(image_path_list=image_list,
                                                      data_root=data_root,
                                                      transform=model_out['test_transform'])
            print(out['num_images'])
            image_size = out['image_size']
            images_preprocessed = out['images_preprocessed']
        else:
            if class_idx not in image_group:
                print(f'Class {class_idx} not in image group. Skipping...')
                continue
            image_list = image_group[class_idx]
            out = ceh.select_class_and_load_images(image_path_list=image_list,
                                                   data_root=data_root,
                                                   transform=model_out['test_transform'])
            image_size = out['image_size']
            patch_size = param_dicts['feature_extraction_params']['patch_size']
            images_preprocessed = ceh.patchify_images(out['images_preprocessed'], patch_size, strides=None)

        for mi in range(len(fe_outs)):
            fe_out = fe_outs[mi]
            concepts_folder = concepts_folders[mi]
            output_dir = output_dirs[mi]

            # Extract model activations
            activations = _batch_inference(backbones[mi], images_preprocessed, batch_size=64,
                                           resize=image_size,
                                           device=param_dicts1['device'])

            layer = fe_out['layer_names'][-1]
            concepts = load_concepts(concepts_folder, layer, class_idx)
            if concepts is None:
                print(f'{class_idx} Concepts missing. Skipping...')
                continue

            W = concepts['W']
            U = compute_concept_coefficients(activations, W, method=decomp_methods[mi])
            W = torch.FloatTensor(W)
            U = torch.FloatTensor(U)
            integrated_gradients = concept_integrated_gradients(U.cuda(), W.cuda(), clfs[mi], class_idx, args.steps, activations=activations)
            torch.save(integrated_gradients.cpu(), os.path.join(output_dir, f'{class_idx}.pth'))


if __name__ == '__main__':
    main()