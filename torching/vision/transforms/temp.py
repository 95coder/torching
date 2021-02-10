import torch
import numpy as np
from numpy import random
from PIL import Image

from torchvision import transforms as T
from torchvision.transforms import functional as F

from torching.vision.structures.box_list import BoxList


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, targets=None):
        for t in self.transforms:
            image, targets = t(image, targets)
        return image, targets


class ToTensor(object):
    def __init__(self):
        self._proxy = T.ToTensor()

    def __call__(self, image, targets=None):
        image = self._proxy(image)

        if isinstance(targets, BoxList):
            targets = targets.to(image.device)
        
        return image, targets


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self._proxy = T.Normalize(mean, std, inplace)

    def __call__(self, image, targets=None):
        return self._proxy(image), targets


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self._proxy = T.Resize(size, interpolation)

    def __call__(self, image, targets=None):
        image = self._proxy(image)

        if isinstance(targets, BoxList):
            targets.resize(self.size)

        return image, targets


class HorizontalFlip(object):
    def __call__(self, image, targets=None):
        image = F.hflip(image)

        if isinstance(targets, BoxList):
            targets.hflip()

        return image, targets


class VerticalFlip(object):
    def __call__(self, image, targets=None):
        image = F.vflip(image)

        if isinstance(targets, BoxList):
            targets.vflip()
        return image, targets


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._proxy = HorizontalFlip()

    def __call__(self, image, targets):
        if random.rand() < self.prob:
            return self._proxy(image, targets)
        return image, targets


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._proxy = VerticalFlip()

    def __call__(self, image, targets):
        if random.rand() < self.prob:
            return self._proxy(image, targets)
        return image, targets


class RandomExpand(object):
    scale_options = (0.1, 0.3, 0.5, 0.7, 0.9)

    def __init__(self, scale_options=None):
        self.scale_options = scale_options or scale_options

    def __call__(self, image, targets=None):
        return image, targets


class RandomCrop(object):
    scale_options = (1, 2, 3, 4)

    def __init__(self, scale_options=None):
        self.scale_options = scale_options or scale_options

    def __call__(self, image, targets=None):
        return image, targets


def _get_image_shape(image):
    return image.shape[-2:]