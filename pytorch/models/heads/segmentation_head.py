import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pt_utils import MaskedUpsample


class SceneSegHeadResNet(nn.Module):
    def __init__(self, num_classes, width, base_radius, nsamples):
        """A scene segmentation head for ResNet backbone.

        Args:
            num_classes: class num.
            width: the base channel num.
            base_radius: the base ball query radius.
            nsamples: neighborhood limits for each layer, a List of int.

        Returns:
            logits: (B, num_classes, N)
        """
        super(SceneSegHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(width // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 2, num_classes, kernel_size=1, bias=True))

    def forward(self, end_points):
        features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],
                            end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])
        features = torch.cat([features, end_points['res4_features']], 1)
        features = self.up_conv0(features)

        features = self.up1(end_points['res3_xyz'], end_points['res4_xyz'],
                            end_points['res3_mask'], end_points['res4_mask'], features)
        features = torch.cat([features, end_points['res3_features']], 1)
        features = self.up_conv1(features)

        features = self.up2(end_points['res2_xyz'], end_points['res3_xyz'],
                            end_points['res2_mask'], end_points['res3_mask'], features)
        features = torch.cat([features, end_points['res2_features']], 1)
        features = self.up_conv2(features)

        features = self.up3(end_points['res1_xyz'], end_points['res2_xyz'],
                            end_points['res1_mask'], end_points['res2_mask'], features)
        features = torch.cat([features, end_points['res1_features']], 1)
        features = self.up_conv3(features)

        logits = self.head(features)

        return logits


class MultiPartSegHeadResNet(nn.Module):
    def __init__(self, num_classes, width, base_radius, nsamples, num_parts):
        """A multi-part segmentation head for ResNet backbone.

        Args:
            num_classes: number of different shape types, a int.
            width: the base channel num.
            base_radius: the base ball query radius.
            nsamples: neighborhood limits for each layer, a List of int.
            num_parts: part num for each shape type, a List of int.

        Returns:
            a List of logits for all shapes. [(B, num_parts_i, N)]
        """
        super(MultiPartSegHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        self.num_parts = num_parts
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        self.multi_shape_heads = nn.ModuleList()
        for i in range(num_classes):
            self.multi_shape_heads.append(
                nn.Sequential(nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                              nn.BatchNorm1d(width // 2),
                              nn.ReLU(inplace=True),
                              nn.Conv1d(width // 2, num_parts[i], kernel_size=1, bias=True)))

    def forward(self, end_points):
        features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],
                            end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])
        features = torch.cat([features, end_points['res4_features']], 1)
        features = self.up_conv0(features)

        features = self.up1(end_points['res3_xyz'], end_points['res4_xyz'],
                            end_points['res3_mask'], end_points['res4_mask'], features)
        features = torch.cat([features, end_points['res3_features']], 1)
        features = self.up_conv1(features)

        features = self.up2(end_points['res2_xyz'], end_points['res3_xyz'],
                            end_points['res2_mask'], end_points['res3_mask'], features)
        features = torch.cat([features, end_points['res2_features']], 1)
        features = self.up_conv2(features)

        features = self.up3(end_points['res1_xyz'], end_points['res2_xyz'],
                            end_points['res1_mask'], end_points['res2_mask'], features)
        features = torch.cat([features, end_points['res1_features']], 1)
        features = self.up_conv3(features)

        logits_all_shapes = []
        for i in range(self.num_classes):
            logits_all_shapes.append(self.multi_shape_heads[i](features))

        return logits_all_shapes
