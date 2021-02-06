import torch
import pytest

from torching.models.detection.ssd.model import make_ssd
from torching.models.detection.ssd.default_config import cfg


def test_inference():
    images = torch.rand((2, 3, 300, 300))
    targets = [torch.rand((2, 5)), torch.rand((5, 5))]

    cfg.head.num_classes = 2
    model = make_ssd(cfg)

    model.eval()
    with torch.set_grad_enabled(False):
        detect_out = model(images)
        print('detect_out: ', detect_out)

    # for name, layer in model.named_modules():
    #     print(name, layer)


if __name__ == "__main__":
    # test_inference()
    pytest.main('-s', 'test_ssd_predict.py')