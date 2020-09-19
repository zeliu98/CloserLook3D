import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalAvgPool1d, self).__init__()

    def forward(self, mask, features):
        out = features.sum(-1)
        pcl_num = mask.sum(-1)
        out /= pcl_num[:, None]
        return out


class ClassifierResNet(nn.Module):
    def __init__(self, num_classes, width):
        """A classifier for ResNet backbone.

        Args:
            num_classes: the number of classes.
            width: the base channel num.

        Returns:
            logits: (B, num_classes)
        """
        super(ClassifierResNet, self).__init__()
        self.num_classes = num_classes
        self.pool = MaskedGlobalAvgPool1d()
        self.classifier = nn.Sequential(
            nn.Linear(16 * width, 8 * width),
            nn.BatchNorm1d(8 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8 * width, 4 * width),
            nn.BatchNorm1d(4 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4 * width, 2 * width),
            nn.BatchNorm1d(2 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2 * width, num_classes))

    def forward(self, end_points):
        pooled_features = self.pool(end_points['res5_mask'], end_points['res5_features'])
        return self.classifier(pooled_features)
