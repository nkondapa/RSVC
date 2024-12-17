import pytorch_lightning as pl
import torch
import os
from PIL import Image
import numpy as np


class BinaryConceptClassifierDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_root, class_idx, transform=None, split=None, seed=0):
        self.dataset_root = dataset_root
        self.dataset_root = os.path.join(self.dataset_root, 'train')

        self.class_idx = class_idx

        self.transform = transform
        self.split = split

        self.class_folders = sorted(os.listdir(self.dataset_root))
        self.num_classes = len(self.class_folders)
        self.image_paths = []
        self.labels = []

        for ci, class_folder in enumerate(self.class_folders):
            is_target = int(ci == class_idx)
            class_path = os.path.join(self.dataset_root, class_folder)
            img_files = os.listdir(class_path)
            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                self.image_paths.append(img_path)
                self.labels.append(is_target)

        self.labels = np.array(self.labels)
        self.image_paths = np.array(self.image_paths)

        self.rng = np.random.default_rng(seed)

        self.split_data(np.arange(len(self.labels)), 0.8, split)

    def split_data(self, indices, pct, split):
        # split the dataset into train and test

        indices = np.arange(len(self.labels))
        in_class_indices = indices[self.labels == 1]
        out_class_indices = indices[self.labels == 0]

        train_pct = 0.8
        self.rng.shuffle(in_class_indices)
        self.rng.shuffle(out_class_indices)

        def _split(indices, train_pct):
            train_idx = int(len(indices) * train_pct)
            if split == 'train':
                indices = indices[:train_idx]
            else:
                indices = indices[train_idx:]
            return indices

        in_class_indices = _split(in_class_indices, train_pct)
        out_class_indices = _split(out_class_indices, train_pct)

        indices = np.concatenate([in_class_indices, out_class_indices])

        self.labels = self.labels[indices]
        self.image_paths = self.image_paths[indices]

    def create_balanced_sampler_weights(self):
        sampler_weights = []
        num_ones = self.labels.sum()
        num_zeros = len(self.labels) - num_ones
        for label in self.labels:
            if label == 0:
                sampler_weights.append(1 / num_zeros)
            else:
                sampler_weights.append(1 / num_ones)
        return sampler_weights
    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        img = self.load_image(image_path)

        if self.transform:
            img = self.transform(img)

        return {'image_path': image_path, 'input': img, 'target': label, 'class_idx': self.class_idx}


