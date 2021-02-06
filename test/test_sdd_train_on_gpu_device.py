import logging
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torching.models.detection.ssd.model import make_ssd
from torching.models.detection.ssd.default_config import cfg
from torching.data.voc import VOCDetectionDataset, data_collate_fn
from torching.train import DetectorTrainer

logging.basicConfig(level=logging.INFO)


def test_train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = make_ssd(cfg)
    model.train()
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    num_epochs = 10
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((cfg.input.image_size[0], cfg.input.image_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.input.mean_value,
            std=cfg.input.std_value,
        ),
    ])
    
    train_dataset = VOCDetectionDataset('/data/datasets/images/VOC/VOCdevkit/VOC2007', 'train', transform)
    val_dataset =  VOCDetectionDataset('/data/datasets/images/VOC/VOCdevkit/VOC2007', 'val', transform)

    # Data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collate_fn)

    DetectorTrainer(model,
                    "ssd",
                    optimizer=optimizer,
                    dataloader=train_dataloader,
                    validation_dataloader=val_dataloader,
                    num_epochs=num_epochs, 
                    scheduler=scheduler,
                    device=device)()


if __name__ == "__main__":
    test_train()