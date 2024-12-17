import torch
import numpy as np
import matplotlib.pyplot as plt


class RandomLowPass(torch.nn.Module):

    def  __init__(self, min_threshold_ratio, fixed_threshold_ratio=None):
        super().__init__()
        self.last_threshold_ratio = None
        self.min_threshold_ratio = min_threshold_ratio
        self.fixed_threshold_ratio = fixed_threshold_ratio

    def forward(self, image_tensor):
        """
        Apply a low-pass filter to the image in Fourier space.

        :param image_tensor: Tensor representation of the image.
        :param threshold: The radius of the low pass filter in Fourier space.
        :return: The filtered image tensor.
        """
        if self.fixed_threshold_ratio is None:
            threshold_ratio = np.random.uniform(0.0, 1.0) * (1 - self.min_threshold_ratio) + self.min_threshold_ratio
        else:
            threshold_ratio = self.fixed_threshold_ratio

        self.last_threshold_ratio = threshold_ratio
        # print(threshold_ratio)
        # Apply FFT
        fft_image = torch.fft.fft2(image_tensor)
        fft_image = torch.fft.fftshift(fft_image)

        # Create a mask for the low-pass filter
        h, w = fft_image.shape[-2:]
        center_h, center_w = h // 2, w // 2

        # Calculate distances from the center of the Fourier transform
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        distances = torch.sqrt((Y - center_h) ** 2 + (X - center_w) ** 2)
        threshold = threshold_ratio * distances.max()

        # Apply the mask to the FFT image
        mask = distances <= threshold
        fft_image_filtered = fft_image * mask
        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(np.log(np.abs(fft_image.permute(1, 2, 0).mean(-1))), cmap='gray')
        # axes[1].imshow(np.log(np.abs(fft_image_filtered.permute(1, 2, 0).mean(-1))), cmap='gray')
        # axes[2].imshow(mask.numpy().astype('uint8'), cmap='gray')
        # plt.show()

        # Convert back to image space
        fft_image_filtered_unshifted = torch.fft.ifftshift(fft_image_filtered)
        ifft_image = torch.fft.ifft2(fft_image_filtered_unshifted).real

        return ifft_image


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    path = '/home/nkondapa/Datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    img = Image.open(path)
    img = transforms.ToTensor()(img)
    rlp = RandomLowPass()
    filt_img = rlp(img)
    print(rlp.last_threshold_ratio)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img.permute(1, 2, 0))
    axes[1].imshow(filt_img.permute(1, 2, 0))
    plt.show()
    print('done')