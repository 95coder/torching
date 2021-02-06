import torch
import pytest

from torching.models.detection.ssd.model import make_ssd
from torching.models.detection.ssd.default_config import cfg


def test_inference():
    images = torch.rand((2, 3, 300, 300))
    targets = [torch.rand((2, 5)), torch.rand((5, 5))]

    cfg.head.num_classes = 2
    model = make_ssd(cfg)

    model.train()
    loss = model(images, targets)
    print('loss: ', loss)


if __name__ == "__main__":
    pytest.main('-s', 'test_ssd_forward.py')