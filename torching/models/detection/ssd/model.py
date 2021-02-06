import torch
from torch import nn

from torching.layers.l2_norm import L2Norm

from .backbones.vgg import make_backbone as make_vgg_backbone
from .pyramid import Pyramid
from .multibox import MultiBox
from .head import Head
from .priorbox_generator import PriorboxGenerator
from .detection_out import DetectionOut
from .multibox_loss import MultiboxLoss


class SSD(nn.Module):
    def __init__(self, backbone, pyramid, head):
        super().__init__()
        self.l2_norm = L2Norm(1, 512)

        self.backbone = backbone
        self.pyramid = pyramid
        self.head = head

    def forward(self, x, targets=None):
        features = []
        
        conv4_3_out, conv7_out = self.backbone(x)
        features.append(self.l2_norm(conv4_3_out))
        features.append(conv7_out)

        pyramid_outs = self.pyramid(conv7_out)
        features += pyramid_outs

        return self.head(features, targets)


def make_ssd(cfg):
    # backbone
    if cfg.backbone.arch in ['vgg16', 'vgg19']:
        backbone = make_vgg_backbone(cfg.backbone.arch,
                                     in_channels=cfg.backbone.in_channels,
                                     out_channels=cfg.backbone.out_channels)
    else:
        raise RuntimeError()

    # pyramid
    pyramid = Pyramid(cfg.backbone.out_channels, cfg.pyramid.layer_cfg)

    # head
    multibox = MultiBox(backbone, pyramid, num_classes=cfg.head.num_classes, cfg=cfg.multibox.box_cfg)

    multibox_loss = MultiboxLoss(cfg.head.num_classes,
                                 cfg.multibox_loss.overlap_thresh,
                                 cfg.multibox_loss.neg_pos_ratio,
                                 cfg.multibox_loss.alpha)
                                 
    priorbox_generator = PriorboxGenerator(cfg.priorbox.pyramid_sizes,
                                           cfg.priorbox.min_scale,
                                           cfg.priorbox.max_scale,
                                           cfg.priorbox.aspect_ratios)

    detection_out = DetectionOut(cfg.box_selector.nms_threshold,
                                 cfg.box_selector.top_k, 
                                 cfg.box_selector.confidence_threshold,
                                 cfg.box_selector.keep_top_k)

    head = Head(multibox, multibox_loss, priorbox_generator, detection_out)

    return SSD(backbone, pyramid, head)