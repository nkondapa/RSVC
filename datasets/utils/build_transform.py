import torchvision
import torchvision.transforms as T
import copy
from PIL import ImageDraw
import torch
from datasets.utils.transforms.paste_object_transform import ModifierPaste


def get_transform(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'concept_crop_embeddings' or dataset_name == 'concept_collage_embeddings' or dataset_name == 'topk_concept_collage_embeddings':
        test_transform = None
        transform = None
        preprocessing = None

        return dict(transform=transform, test_transform=test_transform, preprocessing=preprocessing)

    if dataset_name == 'nabirds' or dataset_name == 'stanford_cars' or dataset_name == 'nabirds_stanford_cars':
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        preprocessing = None
        if params.get('use_test_transform_for_train', False):
            transform = copy.deepcopy(test_transform)

        return dict(transform=transform, test_transform=test_transform, preprocessing=preprocessing)

    if dataset_name == 'nabirds_modified':


        train_config = params['synthetic_concept_config']['train']
        test_config = params['synthetic_concept_config']['test']

        mod_transforms = []

        for _config in [train_config, test_config]:
            if _config['concept_type'] == 'square':
                concept_params = _config['concept_params']
                mod_transforms.append(ModifierPaste(shape='square', size=concept_params['size'], color=concept_params['color'],
                                                     position=concept_params['position'], probability=concept_params['probability']))
            else:
                raise NotImplementedError

        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        modified_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            mod_transforms[0],
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        modified_test_transform = T.Compose([
            T.Resize((224, 224)),
            mod_transforms[1],
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        preprocessing = None

        return dict(transform=transform, test_transform=test_transform, preprocessing=preprocessing,
                    modified_transform=modified_transform, modified_test_transform=modified_test_transform)

    if dataset_name == 'imagenet':
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        preprocessing = None

        return dict(transform=transform, test_transform=test_transform, preprocessing=preprocessing)
