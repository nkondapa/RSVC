import os
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

n_files = {
    "train": 1281160, # actually 1281167,but we delete a corrupted file
    "val": 50000,
}


class ImageNetModified(torchvision.datasets.ImageNet):
    def __init__(self, root, split='train', transform=None, target_transform=None, concept_params=None):
        super(ImageNetModified, self).__init__(root, split=split, transform=transform, target_transform=target_transform)
        self.concept_training = concept_params
        self.active_samples = self.samples
        self.concept_params = concept_params
        self.split = split

        if concept_params is not None:

            self.class_samples = {}
            for i, (path, target) in enumerate(self.samples):
                if target not in self.class_samples:
                    self.class_samples[target] = []
                self.class_samples[target].append((path, target))
            self.concept_samples = []
            for class_ind in concept_params['concept_classes']:
                self.concept_samples.extend(self.class_samples[class_ind])

            if concept_params.get('repeat', 1) > 1 and split == 'train':
                self.concept_samples *= concept_params['repeat']

            if concept_params['join_samples'] and split == 'train':
                # add "dataset index" to the concept samples
                self.concept_samples = [(path, target, 0) for i, (path, target) in enumerate(self.concept_samples)]
                self.samples = [(path, target, 1) for i, (path, target) in enumerate(self.samples)]
                self.active_samples = self.concept_samples + self.samples
            else:
                self.active_samples = self.concept_samples

    def set_active(self, mode='classification'):
        if mode == 'classification':
            self.active_samples = self.samples
        elif mode == 'concept':
            self.active_samples = self.concept_samples
        else:
            raise ValueError(f"Unknown mode {mode}")

    def __getitem__(self, index):
        if self.concept_params is not None and self.split == 'train' and self.concept_params['join_samples']:
            path, target, dataset_idx = self.active_samples[index]
        else:
            path, target = self.active_samples[index]
            dataset_idx = 1 if self.concept_params is None else 0

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return dict(input=sample, target=target, path=path, dataset_idx=dataset_idx)

    def __len__(self):
        return len(self.active_samples)


def imagenet(split: str, transforms=None, imagenet_dir = './data/imagenet/'):
    # n_files_found = sum([len(x) for _, _, x in os.walk(os.path.join(imagenet_dir, split))])
    # assert n_files_found >= n_files[split], \
    # f"Imagenet {split} dataset is not complete. Found {n_files_found}, expected {n_files[split]} files."

    return torchvision.datasets.ImageNet(imagenet_dir, split=split, transform=transforms)

def imagenet_modified(split: str, transforms=None, imagenet_dir = './data/imagenet/'):
    return ImageNetModified(imagenet_dir, split=split, transform=transforms)