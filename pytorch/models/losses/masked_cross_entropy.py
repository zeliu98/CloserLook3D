import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, logit, target, mask):
        loss = F.cross_entropy(logit, target, reduction='none')
        loss *= mask
        return loss.sum() / mask.sum()
