import pytest
import cv2
import os
from PIL import Image

from torching.vision import transforms as T
from torching.vision.structures.box_list import BoxList


data_dir = os.path.join(os.path.dirname(__file__), '../../data')


class _TestImageTransform:
    def setup_class(self):
        self.test_img_path = os.path.join(data_dir, 'people1.jpg')
        self.test_numpy_img = cv2.imread(self.test_img_path)
        self.test_pil_img = Image.open(self.test_img_path)
        self.display = False

    def test_rotate(self):
        rotated_img = T.Rotate(90)(self.test_numpy_img)

        if self.display:
            cv2.imshow('rotate', rotated_img)
            cv2.waitKey()
            cv2.destroyWindow('rotate')

    def test_affine(self):
        affined_img = T.Affine(45, center=(0.1, 0.1), scale=0.5)(self.test_numpy_img)

        if self.display:
            cv2.imshow('affine', affined_img)
            cv2.waitKey()
            cv2.destroyWindow('affine')

    def test_random_affine(self):
        affined_img = T.RandomAffine()(self.test_numpy_img)

        if self.display:
            cv2.imshow('random_affine', affined_img)
            cv2.waitKey()
            cv2.destroyWindow('random_affine')


class TestImageBoxListTransform:
    def setup_class(self):
        self.test_img_path = os.path.join(data_dir, 'people1.jpg')
        self.test_numpy_img = cv2.imread(self.test_img_path)
        self.test_box_list = BoxList([[50, 50, 100, 100]], img_size=(500, 319))
        self.display = True

    def test_resize(self):
        image, box_list = T.Resize((300, 300))(self.test_numpy_img, self.test_box_list)
        # if self.display:
        #     cv2.imshow('rotate', image)
        #     cv2.waitKey()
        #     cv2.destroyWindow('rotate')



if __name__ == '__main__':
    pytest.main(['-s', 'test_transform.py'])