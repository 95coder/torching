import logging
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torching.vision.models.detection.ssd.model import make_ssd
from torching.vision.models.detection.ssd.default_config import cfg
from torching.vision.data.voc import VOCDetectionDataset
from torching.common.utils.trainer import BaseTrainer
from torching.common.utils.checkpointer import CheckPointer
from torching.common.utils.logger import FMT_TNLM


import torching.vision.transforms as T


logging.basicConfig(level=logging.INFO, format=FMT_TNLM)


def test_train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = make_ssd(cfg)
    model.train()
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    num_epochs = 10
    batch_size = 1

    transform = T.Compose([
        T.Resize(cfg.input.image_size),
        T.ToTensor(),
        T.Normalize(
            mean=cfg.input.mean_value,
            std=cfg.input.std_value,
        ),
    ])

    # transform2 = T.Compose([
    #     T.RandomAffine(cfg.horizontal_flip_prob),
    #     T.RandomRotation(cfg.horizontal_flip_prob),
    #     T.RandomCrop(cfg.horizontal_flip_prob),
    #     T.RandomHorizontalFlip(cfg.horizontal_flip_prob),
    #     T.RandomVerticalFlip(cfg.vertical_flip_prob),
    #     T.ColorJitter(brightness=cfg.color_jitter.brightness,
    #                     contrast=cfg.color_jitter.contrast,
    #                     saturation=cfg.color_jitter.saturation,
    #                     hue=cfg.color_jitter.hue),
    #     T.Resize(cfg.image_size),
    #     T.ToTensor(),
    #     T.Normalize(
    #         mean=cfg.mean_value,
    #         std=cfg.std_value,
    #     ),
    # ])
    
    # train_dataset = VOCDetectionDataset('/data2/dataset/VOC/VOC2007', 'train', transform)
    # val_dataset =  VOCDetectionDataset('/data2/dataset/VOC/VOC2007', 'val', transform)

    train_dataset = VOCDetectionDataset('//data/datasets/images/VOC/VOCdevkit/VOC2007', 'train', transform)
    val_dataset =  VOCDetectionDataset('/data/datasets/images/VOC/VOCdevkit/VOC2007', 'val', transform)

    # Data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, collate_fn=VOCDetectionDataset.collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=VOCDetectionDataset.collate_fn)

    checkpointer = CheckPointer('ssd', '/data/models/ssd')

    trainer = BaseTrainer(model,
                          "ssd",
                          optimizer=optimizer,
                          dataloader=train_dataloader,
                          validation_dataloader=val_dataloader,
                          num_epochs=num_epochs, 
                          scheduler=scheduler,
                          checkpointer=checkpointer,
                          device=device)

    trainer()


if __name__ == "__main__":
    test_train()