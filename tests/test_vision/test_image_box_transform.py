import pytest
import cv2
import os
from PIL import Image
import numpy as np

from torching.vision import transforms as T
from torching.vision.structures.box import BoxList
from torching.vision.utils.visualize import BoxTargetDisplay
from torching.vision.utils.visualize import BoxListDisplay

from torchvision import transforms as T2


data_dir = os.path.join(os.path.dirname(__file__), '../../data')


class TestImageBoxTransformVisualization(object):
    def setup_class(self):
        test_img_path = os.path.join(data_dir, 'people2.jpg')
        test_img_boxes = [[350, 130, 475, 515], [550, 115, 710, 530]]
        self.test_img_scores = [1, 1]
        self.test_img_labels = [1, 1]

        self.test_pil_img = Image.open(test_img_path)
        self.test_numpy_img = cv2.imread(test_img_path)

        img_h, img_w = self.test_numpy_img.shape[:2]
        self.test_boxlist = BoxList(test_img_boxes, (img_w, img_h))

    def test_normal(self):
        image = self.test_numpy_img.copy()
        BoxListDisplay(self.test_boxlist, image, win_name='Normal').show()

    def test_resize(self):
        image, boxlist = T.Resize((300, 300))(self.test_numpy_img, self.test_boxlist)
        BoxListDisplay(boxlist, image, win_name='Resize((300, 300))').show()

    def test_rotate(self):
        image, boxlist = T.Rotate(30)(self.test_numpy_img, self.test_boxlist)
        BoxListDisplay(boxlist, image, win_name='Rotate(30)').show()

    def test_random_affine(self):
        image, boxlist = T.RandomAffine()(self.test_numpy_img, self.test_boxlist)
        BoxListDisplay(boxlist, image, win_name='RandomAffine').show()


if __name__ == '__main__':
    pytest.main(['-s', 'test_image_box_transform.py'])