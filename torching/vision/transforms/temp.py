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

    def __call__(self, img, targets=None):
        for t in self.transforms:
            img, targets = t(img, targets)
        return img, targets


class ToTensor(object):
    def __init__(self):
        self._proxy = T.ToTensor()

    def __call__(self, img, targets=None):
        img = self._proxy(img)

        if isinstance(targets, BoxList):
            targets = targets.to(img.device)
        
        return img, targets


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self._proxy = T.Normalize(mean, std, inplace)

    def __call__(self, img, targets=None):
        return self._proxy(img), targets


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self._proxy = T.Resize(size, interpolation)

    def __call__(self, img, targets=None):
        img = self._proxy(img)

        if isinstance(targets, BoxList):
            targets.resize(self.size)

        return img, targets


class HorizontalFlip(object):
    def __call__(self, img, targets=None):
        img = F.hflip(img)

        if isinstance(targets, BoxList):
            targets.hflip()

        return img, targets


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._proxy = HorizontalFlip()

    def __call__(self, img, targets):
        if random.rand() < self.prob:
            return self._proxy(img, targets)
        return img, targets



class Expand(object):
    def __call__(self, image, targets=None):
        return image, targets

class Crop(object):
    def __call__(self, image, targets=None):
        return image, targets


class Rotate(object):
    def __call__(self, image, targets=None):
        return image, targets


def _get_image_shape(img):
    return img.shape[-2:]