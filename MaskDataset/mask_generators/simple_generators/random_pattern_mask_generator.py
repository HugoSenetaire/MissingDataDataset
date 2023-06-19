
from ..abstract_mask_generator import AbstractGenerator

import numpy as np
import torch

from torchvision import transforms
from PIL import Image


class RandomPattern(AbstractGenerator):
    """
    Reproduces "random pattern mask" for inpainting, which was proposed in
    Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T.,
    & Efros, A. A. Context Encoders: Feature Learning by Inpainting.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1604.07379
    This code is based on lines 273-283 and 316-330 of Context Encoders
    implementation:
    https://github.com/pathak22/context-encoder/blob/master/train_random.lua
    The idea is to generate small matrix with uniform random elements,
    then resize it using bicubic interpolation into a larger matrix,
    then binarize it with some threshold,
    and then crop a rectangle from random position and return it as a mask.
    If the rectangle contains too many or too few ones, the position of
    the rectangle is generated again.
    The big matrix is resampled when the total number of elements in
    the returned masks times update_freq is more than the number of elements
    in the big mask. This is done in order to find balance between generating
    the big matrix for each mask (which is involves a lot of unnecessary
    computations) and generating one big matrix at the start of the training
    process and then sampling masks from it only (which may lead to
    overfitting to the specific patterns).
    """
    def __init__(self, max_size=10000, resolution=0.06,
                 density=0.25, update_freq=1, seed=239, accross_channel=True):
        """
        Args:
            max_size (int):      the size of big binary matrix
            resolution (float):  the ratio of the small matrix size to
                                 the big one. Authors recommend to use values
                                 from 0.01 to 0.1.
            density (float):     the binarization threshold, also equals
                                 the average ones ratio in the mask
            update_freq (float): the frequency of the big matrix resampling
            seed (int):          random seed
        """
        super().__init__(accross_channel=accross_channel)
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        """
        Resamples the big matrix and resets the counter of the total
        number of elements in the returned masks.
        """
        low_size = int(self.resolution * self.max_size)
        low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size)) * 255
        low_pattern = torch.from_numpy(low_pattern.astype('float32'))
        pattern = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.max_size, Image.BICUBIC),
                        transforms.ToTensor(),
        ])(low_pattern[None])[0]
        pattern = torch.lt(pattern, self.density).byte()
        self.pattern = pattern.byte()
        self.points_used = 0

    def masking_rule(self, batch, density_std=0.05):
        """
        data is supposed to have shape [num_objects x num_channels x
        x width x height].
        Return binary mask of the same shape, where for each object
        the ratio of ones in the mask is in the open interval
        (self.density - density_std, self.density + density_std).
        The less is density_std, the longer is mask generation time.
        For very small density_std it may be even infinity, because
        there is no rectangle in the big matrix which fulfills
        the requirements.
        """
        assert len(batch['data'].shape) == 4, "RandomPatternGenerator only supports 4D tensors, ie images with channel included"
        batch_size, num_channels, width, height = batch['data'].shape
        res = torch.zeros_like(batch['data'], device=batch['data'].device)
        idx = list(range(batch_size))
        while idx:
            nw_idx = []
            x = self.rng.randint(0, self.max_size - width + 1, size=len(idx))
            y = self.rng.randint(0, self.max_size - height + 1, size=len(idx))
            for i, lx, ly in zip(idx, x, y):
                res[i] = self.pattern[lx: lx + width, ly: ly + height][None]
                coverage = float(res[i, 0].mean())
                if not (self.density - density_std <
                        coverage < self.density + density_std):
                    nw_idx.append(i)
            idx = nw_idx
        self.points_used += batch_size * width * height
        if self.update_freq * (self.max_size ** 2) < self.points_used:
            self.regenerate_cache()
        res = 1-res # Invert the mask because this code was created with 1 for missing data
        return res
