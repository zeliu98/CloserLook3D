import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiShapeCrossEntropy(nn.Module):
    def __init__(self, num_classes):
        super(MultiShapeCrossEntropy, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = F.cross_entropy(logits, pl)
            losses += loss
            for isl in range(self.num_classes):
                if isl == sl:
                    continue
                losses += 0.0 * logits_all_shapes[isl][i].sum()
        return losses / batch_size
