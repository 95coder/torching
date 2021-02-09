import torch
import torch.functional as F
from torch import nn


class L2Norm(nn.Module):
    def __init__(self, scale, channels):
        super().__init__()
        self.gamma = scale
        self.channels = channels
        self.weight = nn.Parameter(torch.Tensor(channels))

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-12
        out = x.div(norm) * self.weight[None, :, None, None]
        return out