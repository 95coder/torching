import torch


class SegMaskList:
    def __init__(self, 
                 masks, 
                 img_size,
                 device=None):
        self.masks = masks
        self.img_size = img_size

    def resize(self, size):
        raise NotImplementedError

    def crop(self, region):
        raise NotImplementedError

    def expand(self, ratios):
        raise NotImplementedError