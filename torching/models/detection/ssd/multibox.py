import torch
from torch import nn

from torching.models.detection.ssd.box import BoxCoder
from torching.models.detection.ssd.box import cxcywh_to_xyxy
from torching.models.detection.ssd.box import nms


_multibox_cfg = [4, 6, 6, 6, 4, 4]  # 各尺度特征图的候选框的个数


class MultiBox(nn.Module):
    def __init__(self, backbone, pyramid, num_classes, cfg=_multibox_cfg):
        super().__init__()
        self.backbone = backbone
        self.pyramid = pyramid
        self.num_classes = num_classes
        self.loc_layers, self.conf_layers = self._make_layers(backbone, pyramid, num_classes, cfg)

    def _make_layers(self, backbone, pyramid, num_classes, box_cfg):
        source_layers = []

        for layer in backbone.feature_out_layers():
            """
            layers:
                conv4_3: (None, 512, 38, 38)
                conv7: (None, 1024, 19, 19)
            """
            source_layers.append(layer)

        for layer in pyramid.feature_out_layers():
            """
            layers:
                conv8_2: (None, 512, 10, 10)
                conv9_2: (None, 256, 5, 5)
                conv10_2: (None, 256, 3, 3)
                conv11_2: (None, 256, 1, 1)
            """
            source_layers.append(layer)

        loc_layers = []
        conf_layers = []
        for k, layer in enumerate(source_layers):
            loc_layers += [nn.Conv2d(layer.out_channels, box_cfg[k] * 4, kernel_size=3, stride=1, padding=1)]
            conf_layers += [nn.Conv2d(layer.out_channels, box_cfg[k] * num_classes, kernel_size=3, stride=1, padding=1)]
        
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    def forward(self, features):
        loc_data = []
        for layer, feature in zip(self.loc_layers, features):
            loc = layer(feature)  # loc_1: (batch_size, 16, 38, 38)
            loc = loc.permute(0, 2, 3, 1).contiguous()  # loc_1: (batch_size, 38, 38, 16)
            loc = loc.view(loc.size(0), -1)
            loc_data.append(loc)
        loc_logits = torch.cat(loc_data, dim=1)
        loc_logits = loc_logits.view(loc.size(0), -1, 4)  # (batch_size, (38**2 * 4 + 19**2 * 6 + 10**2 * 6 + 5**2 * 6 + 3**2 * 4 + 1**2 * 4), 4) -> (batch_size, 8732, 4)
        # print('loc_logits.shape: ', loc_logits.shape)

        conf_data = []
        for layer, feature in zip(self.conf_layers, features):
            conf = layer(feature)  # conf_1: (batch_size, 40, 38, 38)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1)
            conf_data.append(conf)
        conf_logits = torch.cat(conf_data, dim=1)
        conf_logits = conf_logits.view(loc.size(0), -1, self.num_classes)  # (batch_size, 8732, num_classes)
        # print('conf_logits.shape: ', conf_logits.shape)

        logits = torch.cat((loc_logits, conf_logits), dim=-1)  # (batch_size, 8732, 4 + num_classes)
        # print('logits.shape: ', logits.shape)
        return logits


class BoxSelector(nn.Module):
    def __init__(self, nms_threshold=0.5, top_k=400, confidence_threshold=0.5, keep_top_k=200):
        super().__init__()

        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.keep_top_k = keep_top_k

        self.box_coder = BoxCoder()        

    def forward(self, predictions, priors):
        batch_size = predictions.size(0)
        num_classes = 2

        priorboxs = priors[:, :4]

        outputs = torch.tensor([])

        for b in range(batch_size):
            pred_locs = predictions[b, :, :4]
            pred_confs = predictions[b, :, 4:]
            
            pred_boxes = self.box_coder.decode(pred_locs, priorboxs)
            
            objs = torch.tensor([])
            
            scores, labels = torch.max(pred_confs, dim=1)

            scores_gt_idxs = torch.where(scores > self.confidence_threshold)[0].view(-1)

            boxes = pred_boxes[scores_gt_idxs, :4]
            scores = scores[scores_gt_idxs]
            labels = labels[scores_gt_idxs]

            keep, count = nms(boxes, scores, self.nms_threshold, self.top_k)

            objs = torch.cat([labels[keep].view(-1, 1), scores[keep].view(-1, 1), boxes[keep]], dim=1)

            scores = objs[:, 1]
            v, idx = scores.sort(0)
            keep_idx = idx[:self.keep_top_k]

            keep_objs = objs[keep_idx, :]

            outputs = torch.cat([outputs, keep_objs.unsqueeze(0)], dim=0)

        return outputs