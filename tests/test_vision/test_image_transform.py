import pytest
import cv2
import os
from PIL import Image

from torching.vision import transforms as T
from torching.vision.structures import BoxList


data_dir = os.path.join(os.path.dirname(__file__), '../../data')


class TestImageTransform:
    def setup_class(self):
        self.test_img_path = os.path.join(data_dir, 'people1.jpg')
        self.test_numpy_img = cv2.imread(self.test_img_path)
        self.test_pil_img = Image.open(self.test_img_path)
        self.display = False

    def test_rotate(self):
        rotated_img = T.Rotate(90)(self.test_numpy_img)

    def test_affine(self):
        affined_img = T.Affine(45, center=(0.1, 0.1), scale=0.5)(self.test_numpy_img)

    def test_random_affine(self):
        affined_img = T.RandomAffine()(self.test_numpy_img)


if __name__ == '__main__':
    pytest.main(['-s', 'test_image_transform.py'])