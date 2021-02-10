from torching.vision.structures.box_list import BoxList
import pytest
import torch


class TestBoxList:
    def setup_class(self):
        self.bl1 = BoxList([[30, 30, 50, 50], [40, 40, 80, 99]], (100, 100))
        self.bl2 = BoxList([[40, 40, 60, 80], [0, 0, 100, 100]], (100, 100))

    def test_iou(self):
        iou = self.bl1.iou(self.bl2)
        # print('iou: ', iou)

    def test_percent_coords(self):
        assert self.bl1.allclose(self.bl1.percent_coords().absolute_coords().to())

    def test_translate(self):
        assert self.bl1.allclose(self.bl1.translate(10, 10).translate(-10, -10).to())

    def test_expand(self):
        assert self.bl1.allclose(self.bl1.expand([2, 2, 2, 2]).expand([0.5, 0.5, 0.5, 0.5]).to())

    def test_rotate90(self):
        print('self.bl1.rotate90(): ', self.bl1.rotate90())


if __name__ == '__main__':
    pytest.main(['-s', 'test_boxlist.py'])