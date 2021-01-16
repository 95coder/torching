import torch
from torchsummary import summary
from torching.models.detection.ssd.model import make_ssd
from torching.models.detection.ssd.default_config import cfg

model = make_ssd(cfg)

images = torch.rand((1, 3, 300, 300))
targets = torch.rand((1, 2, 14))

# model.eval()
# predictions = model(images)
# print(predictions.shape)

model.train()
loss = model(images, targets)
print('loss: ', loss)