import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

'''
This dataset supports a simple regression task 
where the model is trained to predict the model id from embedding of a concept crop (or a set of concept crops, future work).

'''


class ConceptCropEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, transform=None, split=None, seed=0):
        self.dataset_root = dataset_root
        model_folders = os.listdir(dataset_root)

        labels = []
        embedding_paths = []
        for mi, model_folder in enumerate(model_folders):
            label = mi
            model_path = os.path.join(self.dataset_root, model_folder)
            class_folders = os.listdir(model_path)

            for class_folder in class_folders:
                class_path = os.path.join(model_path, class_folder)
                concept_folders = os.listdir(class_path)
                for concept_folder in concept_folders:
                    concept_crop_path = os.path.join(class_path, concept_folder)
                    crop_files = os.listdir(concept_crop_path)
                    for crop_file in crop_files:
                        crop_file_path = os.path.join(concept_crop_path, crop_file)
                        embedding_paths.append(crop_file_path)
                        labels.append(label)

        self.embedding_paths = np.array(embedding_paths)
        self.labels = np.array(labels)
        self.num_classes = len(np.unique(self.labels))
        self.transform = transform
        self.embedding_dim = np.load(self.embedding_paths[0]).shape[0]

        if split is not None:
            # split the data using the seed
            rng = np.random.default_rng(seed)
            inds = np.arange(len(self.embedding_paths))
            train_num = int(len(inds) * 0.8)
            sel_inds = rng.choice(inds, train_num, replace=False)
            mask = np.zeros(len(inds), dtype=bool)
            mask[sel_inds] = True
            self.embedding_paths = self.embedding_paths[mask] if split == 'train' else self.embedding_paths[~mask]
            self.labels = self.labels[mask] if split == 'train' else self.labels[~mask]

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):

        embedding_path = self.embedding_paths[idx]
        embedding = torch.FloatTensor(np.load(embedding_path))
        label = self.labels[idx]

        return {'input': embedding, 'target': label}


if __name__ == '__main__':
    dataset = ConceptCropEmbeddingsDataset(dataset_root='/home/nkondapa/PycharmProjects/ConceptBook/craft/concept_crop_embeddings/clip', transform=transforms.ToTensor(), split='train', seed=0)
    # print(dataset[60000])
    print(len(dataset))
