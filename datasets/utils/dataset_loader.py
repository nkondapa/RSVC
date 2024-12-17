import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from datasets.concept_crop_embeddings import ConceptCropEmbeddingsDataset
from datasets.concept_crop_collage_embeddings import ConceptCollageEmbeddingsDataset
from datasets.topk_concept_crop_collage_topk_embeddings import TopKConceptCollageEmbeddingsDataset
from datasets.binary_concept_classifier_dataset import BinaryConceptClassifierDataset
from datasets.utils.build_transform import get_transform
from datasets.nabirds import NABirds
from datasets.nabirds_stanford_cars import NABirdsStanfordCars
from datasets.imagenet import ImageNetModified
from torch.utils.data.sampler import WeightedRandomSampler

def get_dataset(params):

    dataset_name = params['dataset_name']

    if dataset_name == 'nabirds':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        class_list = params.get('class_list', None)

        train_dataset = NABirds(root=data_root, transform=transform, train=True, class_list=class_list)
        test_dataset = NABirds(root=data_root, transform=test_transform, train=False, class_list=class_list)
        dataset = NABirds(root=data_root, transform=transform, train=None, class_list=class_list)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'nabirds_modified':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])

        transform = transform_dict['transform']
        modified_transform = transform_dict['modified_transform']
        test_transform = transform_dict['test_transform']
        modified_test_transform = transform_dict['modified_test_transform']
        preprocessing = transform_dict['preprocessing']

        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        modified_params = {'transform': modified_transform, 'modified_classes': params['transform_params']['synthetic_concept_config']['train']['concept_params']['classes']}
        train_dataset = NABirds(root=data_root, transform=transform, train=True, modified_params=modified_params)
        modified_test_params = {'transform': modified_test_transform, 'modified_classes': params['transform_params']['synthetic_concept_config']['test']['concept_params']['classes']}
        test_dataset = NABirds(root=data_root, transform=test_transform, train=False, modified_params=modified_test_params)
        dataset = NABirds(root=data_root, transform=transform, train=None, modified_params=modified_params)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'imagenet':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        concept_params = params.get('concept_params', None)
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        # train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
        # dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        train_dataset = ImageNetModified(root=data_root, split='train', transform=transform, concept_params=concept_params)
        test_dataset = ImageNetModified(root=data_root, split='val', transform=test_transform, concept_params=concept_params)
        dataset = train_dataset + test_dataset

        num_classes = len(train_dataset.classes)

        train_sampler = None

        if params.get('oversample_classes', None):
            factor = params['oversample_factor']
            sample_weights = []
            for sample in train_dataset.samples:
                path, label = sample
                if label in params['oversample_classes']:
                    sample_weights.append(factor)
                else:
                    sample_weights.append(1)
            generator = torch.Generator().manual_seed(seed)
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
            shuffle_train = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'stanford_cars':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        concept_params = params.get('concept_params', None)
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)

        # train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
        # dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
        train_dataset = torchvision.datasets.StanfordCars(root=data_root, split='train', transform=transform)
        test_dataset = torchvision.datasets.StanfordCars(root=data_root, split='test', transform=test_transform)
        dataset = train_dataset + test_dataset

        num_classes = len(train_dataset.classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes,
                    transform=transform, preprocessing=preprocessing)

    if dataset_name == 'nabirds_stanford_cars':
        data_root = params['data_root']
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        seed = params.get('seed', 0)

        transform_dict = get_transform(params['transform_params'])
        transform = transform_dict['transform']
        test_transform = transform_dict['test_transform']
        preprocessing = transform_dict['preprocessing']
        shuffle_train = params.get('shuffle_train', True)
        shuffle_test = params.get('shuffle_test', False)
        class_list = params.get('class_list', None)

        train_dataset = NABirdsStanfordCars(root=data_root, transform=transform, train=True, class_list=class_list)
        test_dataset = NABirdsStanfordCars(root=data_root, transform=test_transform, train=False, class_list=class_list)
        dataset = NABirdsStanfordCars(root=data_root, transform=transform, train=None, class_list=class_list)

        num_classes = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        # transform split is for visualization and is usually paired with no_transform
        # (we separate applying the transform from loading the image)
        return dict(train_loader=train_loader, test_loader=test_loader, all_loader=all_loader,
                    train_dataset=train_dataset, test_dataset=test_dataset, all_dataset=dataset,
                    num_classes=num_classes,
                    transform=transform, preprocessing=preprocessing)