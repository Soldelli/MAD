import torch
from torch.functional import F

class TanLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d):
        mask = self.mask2d
        ious2d = self.scale(ious2d).clamp(0, 1)

        return F.binary_cross_entropy_with_logits(
            scores2d.masked_select(mask),
            ious2d.masked_select(mask),
            )