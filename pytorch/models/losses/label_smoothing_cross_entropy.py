import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing_ratio=0.2):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss
