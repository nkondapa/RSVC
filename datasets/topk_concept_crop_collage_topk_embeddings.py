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


class TopKConceptCollageEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, raw_concepts_root, topk=5, transform=None, split=None, seed=0):
        self.dataset_root = dataset_root
        model_folders = os.listdir(dataset_root)

        labels = []
        embedding_paths = []
        for mi, model_folder in enumerate(model_folders):
            label = mi
            model_path = os.path.join(self.dataset_root, model_folder)
            class_folders = os.listdir(model_path)
            for class_folder in class_folders:
                raw_model_path = os.path.join(raw_concepts_root, model_folder, 'concepts', class_folder)
                importances = np.load(os.path.join(raw_model_path, 'importances.npy'))
                class_path = os.path.join(model_path, class_folder)
                concept_folders = os.listdir(class_path)

                sorted_cf = []
                for concept_folder in concept_folders:
                    cid = int(concept_folder.split('_')[-1])
                    sorted_cf.append((concept_folder, importances[cid]))

                sorted_cf = sorted(sorted_cf, key=lambda x: x[1], reverse=True)
                tmp = []
                for k, (concept_folder, imp) in enumerate(sorted_cf):
                    concept_crop_path = os.path.join(class_path, concept_folder)
                    crop_files = [os.path.join(concept_crop_path, crop_file) for crop_file in os.listdir(concept_crop_path)]
                    tmp.extend(crop_files)
                    if k == topk - 1:
                        break

                embedding_paths.append(np.array(tmp))
                labels.append(label)

        self.embedding_paths = np.array(embedding_paths)
        self.labels = np.array(labels)
        self.num_classes = len(np.unique(self.labels))
        self.transform = transform
        self.collage_size = len(self.embedding_paths[0])
        self.embedding_dim = np.load(self.embedding_paths[0][0]).shape[0]

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

        embedding_paths = self.embedding_paths[idx]
        tmp = []
        for embedding_path in embedding_paths:
            embedding = torch.FloatTensor(np.load(embedding_path))
            tmp.append(embedding)

        embedding = torch.stack(tmp)
        label = self.labels[idx]

        return {'input': embedding, 'target': label}


if __name__ == '__main__':
    dataset = TopKConceptCollageEmbeddingsDataset(dataset_root='/home/nkondapa/PycharmProjects/ConceptBook/craft/concept_crop_embeddings/clip',
                                                  raw_concepts_root='/home/nkondapa/PycharmProjects/ConceptBook/craft/raw',
                                                  transform=transforms.ToTensor(), split='train', seed=0)
    print(dataset[2200])
    print(dataset[2200]['input'].shape)
    print(len(dataset))
    print(dataset.num_classes, dataset.collage_size, dataset.embedding_dim)