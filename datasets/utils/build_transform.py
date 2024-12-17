import torchvision
import torchvision.transforms as T
import copy
from datasets.utils.transforms import RandomLowPass, DiffusionNoise
from PIL import ImageDraw
import torch


def get_transform(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'concept_crop_embeddings' or dataset_name == 'concept_collage_embeddings' or dataset_name == 'topk_concept_collage_embeddings':
        test_transform = None
        transform = None
        preprocessing = None

        return dict(transform=transform, test_transform=test_transform, preprocessing=preprocessing)

    if dataset_name == 'binary_concept_dataset':
        crop_size = params['crop_size']
        transform = T.Compose([
            T.RandomResizedCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        class ModifierPaste:
            def __init__(self, shape='square', size=18, color='magenta', position='top_left', probability=0.7):
                self.shape = shape
                self.size = size
                self.color = color
                self.position = position
                self.probability = probability

                if self.position == 'top_left':
                    base_x = 20
                    base_y = 20
                elif self.position == 'random':
                    base_x = torch.randint(0, 224 - self.size, (1,)).item()
                    base_y = torch.randint(0, 224 - self.size, (1,)).item()
                else:
                    raise NotImplementedError

                if self.shape == 'square':
                    self.img_draw_params = {'xy': (
                    (base_x, base_y), (base_x + self.size, base_y), (base_x + self.size, base_y + self.size),
                    (base_x, base_y + self.size)),
                                            'fill': self.color,
                                            'width': 2}
                else:
                    raise NotImplementedError

            def update_params(self):
                if self.color == 'random':
                    self.img_draw_params['fill'] = (torch.randint(0, 255, (1,)).item(), torch.randint(0, 255, (1,)).item(), torch.randint(0, 255, (1,)).item())
                if self.position == 'random':
                    base_x = torch.randint(0, 224 - self.size, (1,)).item()
                    base_y = torch.randint(0, 224 - self.size, (1,)).item()
                    self.img_draw_params['xy'] = (
                    (base_x, base_y), (base_x + self.size, base_y), (base_x + self.size, base_y + self.size),
                    (base_x, base_y + self.size))

            def __call__(self, img):
                tmp = torch.rand(1)
                if self.probability > 0 and tmp < self.probability:
                    self.update_params()
                    draw = ImageDraw.Draw(img)
                    if self.shape == 'square':

                        draw.polygon(**self.img_draw_params)

                return img

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
