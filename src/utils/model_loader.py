import timm
from timm.data import resolve_data_config, resolve_model_data_config, create_transform
# from timm.data.transforms_factory import create_transform
from torchvision import transforms
import torch.nn as nn
import torch
from src.utils.transformer import WrappedBlock
from models.classifier import ClassificationModel
from datasets.utils.build_transform import get_transform
import copy
from timm.models import ResNet, VisionTransformer


def default_split_points(model_name):
    print('Using default split point for model:', model_name)
    if model_name in ['resnet18.a3_in1k', 'resnet34.a3_in1k', 'resnet50.a3_in1k', 'resnet101.a3_in1k', 'resnet152.a3_in1k']:
        return -2
    elif model_name == 'nf_resnet50.ra2_in1k':
        return 4
    elif model_name == 'resnet18':
        return -2
    elif model_name == 'vit_base_patch16_384.orig_in21k_ft_in1k':
        return -1
    else:
        return -1


def split_model(model, params=None):
    if params is None:
        params = {}
    if isinstance(model, ResNet):
        split_before_gap = params.get('split_before_gap', False)
        if not split_before_gap:
            fc = copy.deepcopy(model.fc)
            backbone = model
            backbone.fc = nn.Identity()
        elif split_before_gap:
            fc = copy.deepcopy(model.fc)
            backbone = model
            backbone.fc = nn.Identity()
            backbone.global_pool = nn.Identity()

        return backbone, fc

    elif isinstance(model, VisionTransformer):
        split_before_final_layer_norm = params.get('split_before_final_layer_norm', True)
        prep_for_ensembling = params.get('prep_for_ensembling', False)
        if split_before_final_layer_norm:
            head = torch.nn.Sequential(copy.deepcopy(model.norm), copy.deepcopy(model.fc_norm),
                                        copy.deepcopy(model.head))
            backbone = model
            backbone.norm = nn.Identity()
            backbone.fc_norm = nn.Identity()
            backbone.head = nn.Identity()
        else:
            head = copy.deepcopy(model.head)
            backbone = model
            backbone.head = nn.Identity()
        if prep_for_ensembling:
            backbone.global_pool = None

        return backbone, head


def load_model(model_name, ckpt_path=None, model_type=None, config=None, device='cpu', eval=True):

    if ckpt_path:

        if model_type is None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if 'model.fc.bias' in ckpt['state_dict']:
                num_classes = ckpt['state_dict']['model.fc.bias'].shape[0]
            else:
                num_classes = None
            if num_classes is not None and ckpt['hyper_parameters']['num_classes'] != num_classes:
                lightning_model = ClassificationModel.load_from_checkpoint(ckpt_path, num_classes=num_classes)
            else:
                lightning_model = ClassificationModel.load_from_checkpoint(ckpt_path)
            model = lightning_model.model
        else:
            # loading from pytorch lightning checkpoint (custom trained models)
            model = model_type.load_from_checkpoint(ckpt_path)
            lightning_model = model

        if eval:
            model = model.to(device).eval().requires_grad_(False)
        else:
            model = model.to(device).train()

        transform_dict = get_transform(lightning_model.dataset_params['transform_params'])
        if lightning_model.dataset_params['dataset_name'] == 'nabirds_modified':
            transform = transform_dict['modified_transform']
            test_transform = transform_dict['modified_test_transform']
            print('Using modified transforms for NABirds')
        else:
            transform = transform_dict['transform']
            test_transform = transform_dict['test_transform']


        to_pil = transforms.ToPILImage()

        return {'model': model, 'config': config, 'model_name': model_name, 'transform': transform,
                'test_transform': test_transform, "lightning_model": lightning_model,
                'to_pil': to_pil, 'model_type': model_type}

    else:

        # load from timm library
        model = timm.create_model(model_name, pretrained=True)
        if eval:
            model = model.to(device).eval().requires_grad_(False)
        else:
            model = model.to(device).train()

        # processing
        if config is None:
            config = resolve_model_data_config(model=model, verbose=True)
        transform = create_transform(**config, is_training=True)
        test_transform = create_transform(**config, is_training=False)
        to_pil = transforms.ToPILImage()

        return {'model': model, 'transform': transform, 'test_transform': test_transform,
                'to_pil': to_pil, 'config': config, 'model_name': model_name, 'model_type': 'timm'}


if __name__ == '__main__':
    model_name = 'resnet50.a3_in1k'
    print(model_name)
    out = load_model(model_name)
    for sp in range(4, 9):
        g, h = model_splitter(model_name, out['model'], sp)
        # print(g)
        # print()
        # print(h)
        x = torch.randn(1, 3, 224, 224).to('cuda')
        print('g(x):', g(x).shape)
        print('h(g(x)):', h(g(x)).shape)

    model_name = 'resnet18.a3_in1k'
    out = load_model(model_name)
    print()
    print(model_name)
    for sp in range(4, 9):
        g, h = model_splitter(model_name, out['model'], sp)
        # print(g)
        # print()
        # print(h)
        x = torch.randn(1, 3, 224, 224).to('cuda')
        print('g(x):', g(x).shape)
        print('h(g(x)):', h(g(x)).shape)