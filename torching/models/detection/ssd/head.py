import torch
from torch import nn
import math


class Head(nn.Module):
    def __init__(self, multibox, multibox_loss, priorbox_generator, box_selector):
        super().__init__()

        self.multibox = multibox
        self.multibox_loss = multibox_loss
        self.priorbox_generator = priorbox_generator
        self.box_selector = box_selector
        self.priors = self.priorbox_generator()

    def forward(self, features, targets=None):
        """
        features: tuple of pyramid features
        targets: (batch_size, num_targets, 4 + num_classes)
        """
        predictions = self.multibox(features)

        # if self.training:
        loss = self.multibox_loss(predictions, targets, self.priors)
        return loss
        # else:
        #     detections = self.box_selector(predictions, self.priors)
        # return detections