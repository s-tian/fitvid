import math
import torch
from torchvision.transforms.autoaugment import RandAugment
from torchvision.transforms import RandomResizedCrop, Resize, InterpolationMode


class RandomResizedCropAntialias(torch.nn.Module):
    def __init__(self, output_size, height_minimum_ratio=0.8):
        super().__init__()
        self.output_size = output_size
        self.height_minimum_ratio = height_minimum_ratio
        self.max_wiggle = self.output_size[0] - math.ceil(
            self.height_minimum_ratio * self.output_size[0]
        )  # maximum amount of wiggle in pixels

        self.resize = Resize(
            self.output_size, interpolation=InterpolationMode.BILINEAR, antialias=True
        )

    def forward(self, x):
        x = self.crop(x)
        return self.resize(x)

    def crop(self, x):
        wiggle = torch.randint(low=0, high=self.max_wiggle + 1, size=(1,)).item()
        starts = torch.randint(low=0, high=wiggle + 1, size=(2,))
        crop_size = self.output_size[0] - wiggle
        assert (
            starts[0] + crop_size <= x.shape[-2]
            and starts[1] + crop_size <= x.shape[-1]
        )
        return x[
            ..., starts[0] : starts[0] + crop_size, starts[1] : starts[1] + crop_size
        ]


class FitVidAugment(torch.nn.Module):

    """
    Data augmentation from FitVid paper (https://arxiv.org/pdf/2106.13195.pdf)
    Random cropping + RandAug.
    """

    def __init__(self, image_size=(64, 64), crop_height_min_ratio=0.8):
        super().__init__()
        # RandAugment params taken from Table 9 of the FitVid paper.
        self.rand_augment = RandAugment(num_ops=1, magnitude=5)
        # Scale to at most 0.8 of either dimension of the frame (0.64 of the total area).
        # Always maintain the same aspect ratio.
        self.resized_crop = RandomResizedCrop(
            image_size, scale=(crop_height_min_ratio**2, 1.0), ratio=(1.0, 1.0)
        )
        # self.resized_crop = RandomResizedCropAntialias(image_size, height_minimum_ratio=crop_height_min_ratio)

    def forward(self, obs):
        """
        :param obs: torch Tensor with shape [..., 3, H, W]
        :return: randomly augmented version of obs as done in the FitVid paper
        """
        initial_shape = obs.shape
        # squeeze all but last 3 dimensions of obs
        obs = obs.view(-1, *initial_shape[-3:])
        obs = self.resized_crop(obs)
        obs = self.rand_augment(obs)
        # unsqueeze all but first 3 dimensions of obs
        obs = obs.view(*initial_shape[:-3], *obs.shape[-3:])
        return obs
