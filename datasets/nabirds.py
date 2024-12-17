# from https://github.com/lvyilin/pytorch-fgvc-dataset/

import os
import pandas as pd
import warnings
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, extract_archive
import numpy as np

class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'images'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=None, modified_params=None,
                 class_list=None):
        super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        # dataset_path = os.path.join(root, 'nabirds')
        dataset_path = os.path.join(root)
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.loader = default_loader
        self.train = train

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train is None:
            self.data = self.data
        elif self.train:
            self.data = self.data[self.data.is_training_img == 1]
        elif not self.train:
            self.data = self.data[self.data.is_training_img == 0]
        else:
            raise ValueError('Invalid value for train')

        self.class_list = class_list
        if self.class_list:
            _labels = pd.Series([self.label_map[t] for t in self.data.target])
            mask = np.array(_labels.isin(self.class_list).tolist())
            self.data = self.data[mask]
            # self.data = self.data.reset_index(drop=True)

        self.samples, self.targets = self.get_samples_and_targets()
        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)
        self.num_classes = len(self.label_map)

        self.class_label_map = {v: (k, self.class_names[str(k)]) for k, v in self.label_map.items()}

        if modified_params is not None:
            self.modified_transform = modified_params['transform']
            self.modified_classes = modified_params['modified_classes']
        else:
            self.modified_transform = None
            self.modified_classes = None

    def get_samples_and_targets(self):
        samples = []
        targets = []
        for i in range(self.data.shape[0]):
            sample = self.data.iloc[i]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = self.label_map[sample.target]
            if self.class_list:
                if target not in self.class_list:
                    continue
            samples.append((path, target))
            targets.append(target)
        return samples, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            if self.modified_classes is not None and (target in self.modified_classes or "all" in self.modified_classes):
                img = self.modified_transform(img)
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return dict(input=img, target=target, path=path)


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


if __name__ == '__main__':
    train_dataset = NABirds('/media/nkondapa/SSD2/data/', train=True, download=False)
    test_dataset = NABirds('/media/nkondapa/SSD2/data/', train=False, download=False)
    all_dataset = NABirds('/media/nkondapa/SSD2/data/', train=None, download=False)

    print(len(train_dataset), len(test_dataset), len(all_dataset))