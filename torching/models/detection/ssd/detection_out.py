import torch
from torch import nn
from torching.models.detection.ssd.box import nms
from torching.models.detection.ssd.box import BoxCoder


class DetectionOut(nn.Module):
    def __init__(self,
                 nms_threshold=0.5,
                 top_k=400,
                 confidence_threshold=0.5,
                 keep_top_k=200):
        super().__init__()

        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.keep_top_k = keep_top_k

        self.box_coder = BoxCoder()        

    def forward(self, predictions, priors):
        batch_size = predictions.size(0)

        priorboxs = priors[:, :4]

        outputs = torch.tensor([])
        if predictions.is_cuda:
            outputs = outputs.cuda()

        for b in range(batch_size):
            pred_locs = predictions[b, :, :4]
            pred_confs = predictions[b, :, 4:]
            
            pred_boxes = self.box_coder.decode(pred_locs, priorboxs)
            
            objs = torch.tensor([])
            
            scores, labels = torch.max(pred_confs, dim=1)

            gt_idxs = torch.where(scores > self.confidence_threshold)[0].view(-1)
            boxes = pred_boxes[gt_idxs, :4]
            scores = scores[gt_idxs]
            labels = labels[gt_idxs]

            keep, count = nms(boxes, scores, self.nms_threshold, self.top_k)

            objs = torch.cat([boxes[keep], labels[keep].view(-1, 1).float(), scores[keep].view(-1, 1)], dim=1)

            scores = objs[:, 1]
            v, idx = scores.sort(0)
            keep_idx = idx[:self.keep_top_k]

            keep_objs = objs[keep_idx, :]

            outputs = torch.cat([outputs, keep_objs.unsqueeze(0)], dim=0)

        return outputs