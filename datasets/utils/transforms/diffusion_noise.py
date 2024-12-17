import torch
import numpy as np
import matplotlib.pyplot as plt


class DiffusionNoise(torch.nn.Module):

    def __init__(self, sqrt_alpha=True):
        super().__init__()
        self.sqrt_alpha = sqrt_alpha
        self.last_noise_level = None

    def forward(self, img):
        alpha = np.random.uniform(0.0, 1.0)
        if self.sqrt_alpha:
            alpha = np.sqrt(alpha)
        std = np.sqrt(1 - alpha ** 2)
        noise = torch.randn_like(img) * std
        img = alpha * img + noise
        self.last_noise_level = std
        return img



if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    path = '/home/nkondapa/Datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    img = Image.open(path)
    img = transforms.ToTensor()(img)
    noise = DiffusionNoise()
    noise_img = noise(img)
    print(noise.last_noise_level)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img.permute(1, 2, 0))
    axes[1].imshow(noise_img.permute(1, 2, 0))
    plt.show()
    print('done')