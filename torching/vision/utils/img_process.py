import torch
import numpy as np
import numbers
from collections import Sequence
import cv2
import PIL


INTERPOLATION_NEAREST = 0
INTERPOLATION_LINEAR = 1
INTERPOLATION_BILINER = 2
INTERPOLATION_BICUBIC = 3


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and img.ndim in {2, 3}


def get_numpy_image_size(image):
    if image.ndim == 2:
        img_h, img_w = image.shape
    elif image.ndim == 3:
        img_h, img_w, _ = image.shape
    return (img_w, img_h)


def is_pil_image(img):
    return isinstance(img, PIL.Image.Image)


def get_pil_image_size(image):
    return image.size


def numpy_image_affine_transform(image, matrix, fillcolor, interpolation=INTERPOLATION_NEAREST):
    assert is_numpy_image(image)

    flags = cv2.INTER_LINEAR
    if interpolation == INTERPOLATION_NEAREST:
        flags = cv2.INTER_NEAREST
    elif interpolation == INTERPOLATION_LINEAR:
        flags = cv2.INTER_LINEAR
    elif interpolation == INTERPOLATION_BICUBIC:
        flags = cv2.INTER_CUBIC

    img_size = get_numpy_image_size(image)

    if isinstance(fillcolor, numbers.Number):
        if image.ndim == 2:
            fillcolor = fillcolor
        fillcolor = [fillcolor] * image.shape[-1]

    image = cv2.warpAffine(image, matrix, img_size, 
                           flags=flags, borderValue=fillcolor)
    return image


def pil_image_affine_transform(image, matrix, fillcolor, interpolation=INTERPOLATION_NEAREST):
    assert is_pil_image(image)
    
    resample = PIL.Image.NEAREST
    if interpolation == INTERPOLATION_NEAREST:
        resample = PIL.Image.NEAREST
    elif interpolation == INTERPOLATION_LINEAR:
        resample = PIL.Image.LINEAR
    elif interpolation == INTERPOLATION_BICUBIC:
        resample = PIL.Image.BICUBIC

    img_size = get_pil_image_size(image)

    kwargs = {
        'resample': resample,
        'fillcolor': fillcolor,
    }
    image = image.transform(img_size, PIL.Image.AFFINE, matrix, **kwargs)

    return image


def get_affine_matrix(center, angle, translate, scale, shear):
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    return M