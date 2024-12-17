import os
import numpy as np

def load_imagenet_image_paths():
    path = '../data/imagenet/val/'
    image_folders = os.listdir(path)
    image_paths = []
    for folder in image_folders:
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        image_paths.extend([(os.path.join(folder_path, image), image) for image in images])

    # sort by image name
    image_paths = sorted(image_paths, key=lambda x: x[1])
    # convert to arr
    image_paths = np.array(image_paths)[:, 0]
    return image_paths