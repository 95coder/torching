import torch
import numpy as np
from numpy import random
import PIL
import numbers
import cv2

from torchvision import transforms as T
from torchvision.transforms import functional as F

from torching.vision.structures import BoxList
from torching.vision.utils import img_process


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        
        if target is None:
            return image
        return image, target


class BaseImageTargetTransform(object):
    def __call__(self, image, target=None):
        if img_process.is_numpy_image(image):
            image = self.process_numpy_image(image)
        elif img_process.is_pil_image(image):
            image = self.process_pil_image(image)
        elif torch.is_tensor(image):
            image = self.process_torch_image(image)
        else:
            raise TypeError('Not support transform from `{}`'.format(type(image)))

        if target is not None:
            target = self.process_target(target)

        if target is None:
            return image
        return image, target

    def process_numpy_image(self, image):
        raise TypeError('Not support transform from `{}`'.format(type(image)))

    def process_pil_image(self, image):
        raise TypeError('Not support transform from `{}`'.format(type(image)))

    def process_torch_image(self, image):
        raise TypeError('Not support transform from `{}`'.format(type(image)))

    def process_target(self, target):
        return target


class ToTensor(BaseImageTargetTransform):
    def process_numpy_image(self, image):
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        
        if image.dtype == torch.float32:
            return image
        elif image.dtype == torch.uint8:
            return image.float().div(255)

    def process_pil_image(self, image):
        return F.to_tensor(image)

    def process_torch_image(self, image):
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        return torch.as_tensor(image)


class ToPILImage(BaseImageTargetTransform):
    def process_numpy_image(self, image):
        return F.to_pil_image(image)

    def process_torch_image(self, image):
        return F.to_pil_image(image)


class ToCVImage(BaseImageTargetTransform):
    def process_torch_image(self, image):
        np_img = image.cpu().numpy().transpose((1, 2, 0))
        if image.dtype == torch.float32:
            return np_img.astype(torch.float32)
        elif image.dtype == torch.uint8:
            return np_img.astype(torch.uint8)

    def process_pil_image(self, image):
        # if image.mode == 'F':
        #     return np.asarray(image, np.float32)
        # elif image.mode == 'L':
        #     return np.asarray(image, np.uint8)
        if image.mode == 'BGR':
            return np.asarray(image, np.uint8)
        elif image.mode == 'RGB':
            return np.asarray(image.convert('BGR'), np.uint8)

    def process_numpy_image(self, image):
        return np.asarray(image)


class ToBGR255(BaseImageTargetTransform):
    def process_torch_image(self, image):
        if image.dtype == torch.float32:
            return (image * 255).type(dtype=torch.uint8)
        elif image.dtype == torch.uint8:
            return image

    def process_pil_image(self, image):
        if image.mode == 'BGR':
            return image
        elif image.mode == 'RGB':
            return image.convert('BGR')

    def process_numpy_image(self, image):
        if image.dtype == np.uint8:
            return image
        elif image.dtype == np.float32:
            return (image * 255).astype(np.uint8)


class Normalize(BaseImageTargetTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def process_torch_image(self, image):
        return F.normalize(image, self.mean, self.std)

    def process_numpy_image(self, image):
        image = np.array(image)

        if image.ndim == 2:
            image = image[..., None]

        mean = np.asarray(self.mean, dtype=image.dtype)
        std = np.asarray(self.std, dtype=image.dtype)

        return (image - mean) / std


class Resize(BaseImageTargetTransform):
    def __init__(self, size):
        self.size = tuple(size)

    def process_numpy_image(self, image):
        return cv2.resize(image, self.size, interpolation=None)

    def process_pil_image(self, image):
        return F.resize(image, self.size, interpolation=PIL.Image.BILINEAR)

    def process_target(self, target):
        if isinstance(target, BoxList):
            target = target.resize(self.size)
        return target


class HorizontalFlip(BaseImageTargetTransform):
    def process_pil_image(self, image):
        return F.hflip(image)

    def process_target(self, target):
        if isinstance(target, BoxList):
            target = target.hflip()
        return target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._proxy = HorizontalFlip()

    def __call__(self, image, target=None):
        if random.rand() < self.prob:
            image, target = self._proxy(image, target)

        if target is None:
            return image
        image, target


class VerticalFlip(BaseImageTargetTransform):
    def process_pil_image(self, image):
        return F.vflip(image)

    def process_target(self, target):
        if isinstance(target, BoxList):
            target = target.vflip()
        return target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self._proxy = VerticalFlip()

    def __call__(self, image, target=None):
        if random.rand() < self.prob:
            image, target = self._proxy(image, target)

        if target is None:
            return image
        image, target


# class Expand(object):
#     def __init__(self, scale=None):
#         self.scale = scale

#     def __call__(self, image, target=None):
#         if target is None:
#             return image
#         image, target
        

# class RandomExpand(object):
#     scale_options = (0.1, 0.3, 0.5, 0.7, 0.9)

#     def __init__(self, scale_options=None):
#         self.scale_options = scale_options or scale_options

#     def __call__(self, image, target=None):
#         scale = random.choice(self.scale_options)
#         return Expand(scale)(image, target)


# class Crop(object):
#     def __init__(self, scale=None):
#         self.scale = scale

#     def __call__(self, image, target=None):
#         if target is None:
#             return image
#         image, target


# class RandomCrop(object):
#     scale_options = (1, 2, 3, 4)

#     def __init__(self, scale_options=None):
#         self.scale_options = scale_options or scale_options

#     def __call__(self, image, target=None):
#         scale = random.choice(self.scale_options)
#         return Expand(scale)(image, target)


class Affine(BaseImageTargetTransform):
    def __init__(self, angle, center=None, translate=None, 
                 scale=None, shear=None, fillcolor=0):
        self.angle = angle

        if center is None:
            center = (0.5, 0.5)
        self.center = tuple(center)

        if translate is None:
            translate = (0, 0)
        self.translate = tuple(translate)

        if scale is None:
            scale = 1
        self.scale = scale

        self.shear = shear

        self.fillcolor = fillcolor

    def process_numpy_image(self, image):
        img_w, img_h = img_process.get_numpy_image_size(image)
        
        center = (self.center[0] * img_w, self.center[1] * img_h)
        translate = (self.translate[0] * img_w, self.translate[1] * img_h)

        affine_mat = img_process.get_affine_matrix(center, self.angle, translate,
                                                   self.scale, self.shear)
                                                    
        image = img_process.numpy_image_affine_transform(image, affine_mat, self.fillcolor)
        return image

    def process_pil_image(self, image):
        img_w, img_h = img_process.get_pil_image_size(image)

        center = (self.center[0] * img_w, self.center[1] * img_h)
        translate = (self.translate[0] * img_w, self.translate[1] * img_h)

        affine_mat = img_process.get_affine_matrix(center, self.angle, translate,
                                                   self.scale, self.shear)

        image = img_process.numpy_image_affine_transform(image, affine_mat, self.fillcolor)
        return image

    def process_target(self, target):
        if isinstance(target, BoxList):
            target = target.affine(self.angle, self.center, self.translate, self.scale, self.shear)
        return target


class RandomAffine(object):
    center_range = [0.2, 0.8]
    angle_range = [-30, 30]
    translate_range = [0.0, 0.0]
    scale_range = [0.5, 3]

    def __init__(self,
                 center_range=None,
                 angle_range=None,
                 translate_range=None,
                 scale_range=None,
                 fillcolor=0):

        if angle_range is not None:
            self.angle_range = tuple(angle_range)
        if center_range is not None:
            self.center_range = tuple(center_range)
        if scale_range is not None:
            self.scale_range = tuple(scale_range)

        self.fillcolor = fillcolor

    def __call__(self, image, target=None):
        angle = random.uniform(*self.angle_range)
        center = [random.uniform(*self.center_range), random.uniform(*self.center_range)]
        translate = [random.uniform(*self.translate_range), random.uniform(*self.translate_range)]
        scale = random.uniform(*self.scale_range)

        return Affine(angle, center, translate, scale, fillcolor=self.fillcolor)(image, target)


class Rotate(BaseImageTargetTransform):
    def __init__(self, angle, center=None, fillcolor=0):
        self.angle = angle

        if center is None:
            center = (0.5, 0.5)
        self.center = tuple(center)

        self.fillcolor = fillcolor

    def process_numpy_image(self, image):
        img_w, img_h = img_process.get_numpy_image_size(image)

        center = (self.center[0] * img_w, self.center[1] * img_h)

        affine_mat = img_process.get_affine_matrix(center, self.angle, [0, 0], 1.0, [0, 0])
        image =  img_process.numpy_image_affine_transform(image, affine_mat, self.fillcolor)
        return image

    def process_pil_image(self, image):
        img_w, img_h = img_process.get_pil_image_size(image)

        center = (self.center[0] * img_w, self.center[1] * img_h)

        affine_mat = img_process.get_affine_matrix(center, self.angle, [0, 0], 1.0, [0, 0])
        image =  img_process.pil_image_affine_transform(image, affine_mat, self.fillcolor)
        return image

    def process_target(self, target):
        if isinstance(target, BoxList):
            target = target.rotate(self.angle, self.center)
        return target


class RandomRotate(BaseImageTargetTransform):
    angle_range = (-180, 180)

    def __init__(self,
                 angle_range=None,
                 fillcolor=0):
        if angle_range is not None:
            self.angle_range = angle_range
        self.fillcolor = fillcolor

    def __call__(self, image, target=None):
        angle = random.uniform(*self.angle_range)
        return Rotate(angle, fillcolor=self.fillcolor)(image, target)


class ColorJitter(BaseImageTargetTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self._proxy = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target=None):
        image = self._proxy(image)
        if target is None:
            return image
        image, target