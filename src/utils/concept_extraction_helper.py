import torch
import torchvision.transforms

from src import eval_model
import numpy as np
from PIL import Image
import os
import timm


def group_images(method, params):

    if method == 'craft':
        predictions = params['predictions']
        # group images by their predicted class (CRAFT format)
        pred_label_groups = eval_model.convert_predictions_to_label_groups(predictions)

        # subsample N images per class
        num_images = params['num_images']
        seed = params['seed']
        if num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in pred_label_groups.keys():
                path_list = sorted(pred_label_groups[i])
                rng.shuffle(path_list)
                subsampled_label_groups[i] = path_list[:num_images]

            pred_label_groups = subsampled_label_groups
    else:
        raise ValueError(f'Unknown method: {method}')

    return pred_label_groups

# TODO delete this function after replacing usages
def select_class_and_load_images_v2(image_path_list, data_root, transform, return_raw_images=False, num_image_repeats=1):

    sel_paths = image_path_list
    gt_labels = np.array([path.split('/')[-2] for path in sel_paths])
    basic_transform = None
    if return_raw_images:
        basic_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

    images = []
    transformed_images = []
    for img_path in sel_paths:
        if 'data/stanford_cars' in img_path:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(os.path.join(data_root, img_path.lstrip('/'))).convert('RGB')
        if basic_transform:
            images.append(basic_transform(image))
        transformed_images.append(transform(image))
    images = torch.stack(images, 0) if return_raw_images else None
    images_preprocessed = torch.stack(transformed_images, 0)

    out = {
        'image_paths': sel_paths,
        'gt_labels': gt_labels,
        'images_preprocessed': images_preprocessed,
        'num_images': len(sel_paths),
        'image_size': images_preprocessed.shape[2],
        'images': images,
    }
    return out


def select_class_and_load_images(image_path_list, data_root, transform, return_raw_images=False, num_repeats=1):

    sel_paths = image_path_list
    gt_labels = np.array([path.split('/')[-2] for path in sel_paths])
    basic_transform = None
    if return_raw_images:
        basic_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

    images = []
    transformed_images = []
    image_ids = []
    repeated_image_paths = []
    for ii, img_path in enumerate(sel_paths):
        if 'data/stanford_cars' in img_path:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(os.path.join(data_root, img_path.lstrip('/'))).convert('RGB')

        if basic_transform:
            base_image = basic_transform(image)

        if num_repeats > 1:
            for _ in range(num_repeats):
                if basic_transform:
                    images.append(base_image)
                transformed_images.append(transform(image))
                image_ids.append(ii)
            repeated_image_paths.append(img_path)
        else:
            transformed_images.append(transform(image))

    images = torch.stack(images, 0) if return_raw_images else None
    images_preprocessed = torch.stack(transformed_images, 0)
    image_ids = np.array(image_ids)

    out = {
        'image_paths': repeated_image_paths,
        'gt_labels': gt_labels,
        'images_preprocessed': images_preprocessed,
        'num_images': len(repeated_image_paths),
        'image_size': images_preprocessed.shape[2],
        'images': images,
        'image_ids': image_ids
    }
    return out


def patchify_images(inputs, patch_size, strides):
    assert len(inputs.shape) == 4, "Input data must be of shape (n_samples, channels, height, width)."
    assert inputs.shape[2] == inputs.shape[3], "Input data must be square."

    image_size = inputs.shape[2]

    # extract patches from the input data, keep patches on cpu
    strides = int(patch_size * 0.80)

    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4, 4)
    # for axi, ax in enumerate(axes.flatten()):
    #     ax.axis('off')
    #     ax.imshow(patches[axi].permute(1, 2, 0))
    # plt.show()
    return patches


def load_feature_extraction_layers(model, feature_layer_params):
    out = {}
    feature_layer_version = feature_layer_params['feature_layer_version']

    if isinstance(model, timm.models.resnet.ResNet):
        if feature_layer_version == 'v0':
            out['layer_type'] = 'layer'
            out['layers'] = [model.layer1, model.layer2, model.layer3, model.layer4]
            out['layer_names'] = ['layer1', 'layer2', 'layer3', 'layer4']

            def post_activation_func(x):
                return x.mean((-1, -2))
            out['post_activation_func'] = post_activation_func

        elif feature_layer_version == 'v1':
            layers = []
            layer_names = []
            for name, module in model.named_modules():
                if type(module) == torch.nn.modules.ReLU:
                    layers.append(module)
                    layer_names.append(name)
            out['layers'] = layers
            out['layer_names'] = layer_names
            out['layer_type'] = 'relu'

            def post_activation_func(x):
                return x.mean((-1, -2))
            out['post_activation_func'] = post_activation_func

        else:
            raise ValueError(f'Unknown feature_layer_version: {feature_layer_version}')

    elif isinstance(model, timm.models.vision_transformer.VisionTransformer):
        if feature_layer_version == 'v0':
            out['layer_type'] = 'blmlp'
            out['layers'] = [model.blocks[i] for i in range(len(model.blocks))]
            out['layer_names'] = [f'block{i}' for i in range(len(model.blocks))]
            def post_activation_func(x):
                return x[:, 0, :] # take the first token
            out['post_activation_func'] = post_activation_func
        elif feature_layer_version == 'v1':
            out['layer_type'] = 'norm'
            out['layers'] = [model.norm]
            out['layer_names'] = ['norm']
            def post_activation_func(x):
                return x[:, 0, :]
            out['post_activation_func'] = post_activation_func
        else:
            raise ValueError(f'Unknown feature_layer_version: {feature_layer_version}')
    else:
        raise ValueError(f'Unknown model type: {type(model)}')

    return out

