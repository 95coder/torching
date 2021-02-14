class ImageObject:
    def total(self):
        raise NotImplementedError

    def absolute_coords(self):
        raise NotImplementedError

    def percent_coords(self):
        raise NotImplementedError

    def resize(self, size):
        raise NotImplementedError

    def crop(self, region):
        raise NotImplementedError

    def expand(self, ratios):
        raise NotImplementedError

    def translate(self, x, y):
        raise NotImplementedError

    def vflip(self):
        raise NotImplementedError

    def hflip(self):
        raise NotImplementedError

    def rotate(self, angle, center=None):
        raise NotImplementedError

    def affine(self, angle, center=None, translate=None, scale=1, shear=None):
        raise NotImplementedError