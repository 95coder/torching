import torch
from torchsummary import summary
from torching.models.detection.ssd.model import make_ssd
from torching.models.detection.ssd.default_config import cfg


if __name__ == "__main__":
    images = torch.rand((6, 3, 300, 300))
    targets = torch.rand((6, 2, 6))

    model = make_ssd(cfg)

    with torch.set_grad_enabled(False):
        model.eval()
        # torch.save(model.state_dict(), './test.pth')
        predictions = model(images)
        print(predictions.shape)

    # model.train()
    # loss = model(images, targets)
    # print('loss: ', loss)