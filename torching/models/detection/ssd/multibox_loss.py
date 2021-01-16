import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class BoxCoder:
    def __init__(self):
        """
        box与参考box偏移变换!
        """
    def encode(self, box, reference_box):
        return (box - reference_box) / reference_box

    def decode(self, box, reference_box):
        return (box + 1) * reference_box


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh=0.5, neg_pos_ratio=3, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.box_coder = BoxCoder()

    def forward(self, predictions, targets, priors):
        """
        Argument:
            predictions: 模型预测， 形状为: (batch_size, 8732, 4 + num_classes)
            targets: groundtruths, 形状为: (batch_size, num_targets, 4 + num_classes)
            priors: 先验框，形状为： (8732, 4)

        Returns:
            loss: 损失标量
        """
        assert(predictions.size(0) == targets.size(0))

        loc_p = predictions[:, :, :4]  # (batch_size, num_priors, 4)
        conf_p = predictions[:, :, 4:]   # (batch_size, num_priors, num_classes)
        N = loc_p.size(0)
        num_priors = loc_p.size(1)
        
        # 匹配targets和priors，构建groundtruth
        loc_t, conf_t = self._match_targets_and_priors(targets, priors, self.overlap_thresh) # (batch_size, num_priors, 4), (batch_size, num_priors)

        # groundtruth中大多数都是负样本, 还需要做困难样本挖掘（Hard Negative Mining）.
        # 方法是对confidence loss进行从小到大排序，选择top的neg_pos_ratio * num_pos个负样本.
        pred_probs = conf_p.view(-1, self.num_classes)
        target_labels = conf_t.flatten()

        loss_conf = F.cross_entropy(pred_probs, target_labels, reduce=False)  # probs为one-hot label, labels为ID
        loss_conf = loss_conf.view(N, -1)  # (batch_size, num_priors)
        _, sort_idx = loss_conf.sort(dim=1, descending=True)  # (batch_size, num_priors)
        sort_idx, _ = sort_idx.sort(dim=-1, descending=True)  # (batch_size, num_priors)

        pos_mask = conf_t > 0  # BooleanTensor, (batch_size, num_priors)
        num_pos = pos_mask.sum(dim=1)  # (batch_size,)
        num_neg = self.neg_pos_ratio * num_pos  # (batch_size,)
        num_total = (num_pos + num_neg).clamp(0, num_priors)  # (batch_size,)
        
        neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
        for batch, num in enumerate(num_total):
            neg_mask[batch, num_pos[batch]:num_total[batch]] = True

        loss_loc = self._cal_loc_loss(loc_p, loc_t)
        loss_conf = self._cal_conf_loss(conf_p, conf_t)
        loss = 1. / N * (loss_loc + self.alpha * loss_conf)
        return loss

    def _match_targets_and_priors(self, targets, priors, overlap_thresh):
        """
        根据IOU，匹配targets和priors, 用匹配targets设置的prior的值以构建groundtruth.

        Returns:
            loc_t: 位置groundtruth
            conf_t: 分类groundtruth            
        """
        batch_size = targets.size(0)
        num_priors = priors.size(0)

        loc_t = priors.unsqueeze(0).repeat_interleave(batch_size, dim=0)  # (batch_size, num_priors, 4)
        conf_t = torch.zeros((batch_size, num_priors), dtype=torch.long)  # (batch_size, num_priors)

        for idx in range(batch_size):
            targetboxs = targets[idx, :, :4]  # (num_objs, 4)
            targetlabels = targets[idx, :, 4:]  # (num_objs, num_classes)
            matches = match(targetboxs, priors, overlap_thresh)
            for i, j in matches:
                loc_t[idx, j, :] = targetboxs[idx, i, :]
                conf_t[idx, j] = torch.argmax(targetlabels[idx, i, :]).item()

        return loc_t, conf_t

    def _cal_loc_loss(self, loc_p, loc_t):
        return F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

    def _cal_conf_loss(self, conf_p, conf_t):
        pred_probs = conf_p.view(-1, self.num_classes)
        target_labels = conf_t.flatten()
        return F.cross_entropy(pred_probs, target_labels, reduction='sum')


def match(targetboxs, priorboxs, thresh=0):
    # print('targetboxs.shape: ', targetboxs.shape)
    # print('priorboxs.shape: ', priorboxs.shape)
    iou = jaccard_iou(targetboxs, priorboxs)  # (N, M)
    indices = linear_sum_assignment(iou, maximize=True)

    matches = []
    for i, j in zip(indices[0], indices[1]):
        if iou[i, j] > thresh:
            matches.append([i, j])
    return matches


def jaccard_iou(targetboxs, priorboxs, eplison=1e-5):
    """
    modified from `torchvision.ops.boxes.box_iou`.
    """
    area1 = (targetboxs[:, 2:] - targetboxs[:, :2]).abs().prod(axis=1)  # (N,)
    area2 = (priorboxs[:, 2:] - priorboxs[:, :2]).abs().prod(axis=1)  # (M,)

    lt = torch.max(targetboxs[:, None, :2], priorboxs[:, :2])  # (N, M, 2)
    rb = torch.min(targetboxs[:, None, 2:], priorboxs[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_union = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    iou = area_union / (area1[:, None] + area2 - area_union)  # (N, M)
    return iou + eplison