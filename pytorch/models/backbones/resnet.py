import os
import sys
from ..local_aggregation_operators import LocalAggregation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

import torch.nn as nn
from pt_utils import MaskedMaxPool


class MultiInputSequential(nn.Sequential):

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, radius, nsample, config,
                 downsample=False, sampleDl=None, npoint=None):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        if downsample:
            self.maxpool = MaskedMaxPool(npoint, radius, nsample, sampleDl)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels // bottleneck_ratio, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))
        self.local_aggregation = LocalAggregation(out_channels // bottleneck_ratio,
                                                  out_channels // bottleneck_ratio,
                                                  radius, nsample, config)
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels // bottleneck_ratio, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))

    def forward(self, xyz, mask, features):
        if self.downsample:
            sub_xyz, sub_mask, sub_features = self.maxpool(xyz, mask, features)
            query_xyz = sub_xyz
            query_mask = sub_mask
            identity = sub_features
        else:
            query_xyz = xyz
            query_mask = mask
            identity = features

        output = self.conv1(features)
        output = self.local_aggregation(query_xyz, xyz, query_mask, mask, output)
        output = self.conv2(output)

        if self.in_channels != self.out_channels:
            identity = self.shortcut(identity)

        output += identity
        output = self.relu(output)

        return query_xyz, query_mask, output


class ResNet(nn.Module):
    def __init__(self, config, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        """Resnet Backbone

        Args:
            config: config file.
            input_features_dim: dimension for input feature.
            radius: the base ball query radius.
            sampleDl: the base grid length for sub-sampling.
            nsamples: neighborhood limits for each layer, a List of int.
            npoints: number of points after each sub-sampling, a list of int.
            width: the base channel num.
            depth: number of bottlenecks in one stage.
            bottleneck_ratio: bottleneck ratio.

        Returns:
            A dict of points, masks, features for each layer.
        """
        super(ResNet, self).__init__()

        self.input_features_dim = input_features_dim

        self.conv1 = nn.Sequential(nn.Conv1d(input_features_dim, width // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(width // 2, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))
        self.la1 = LocalAggregation(width // 2, width // 2, radius, nsamples[0], config)
        self.btnk1 = Bottleneck(width // 2, width, bottleneck_ratio, radius, nsamples[0], config)

        self.layer1 = MultiInputSequential()
        sampleDl *= 2
        self.layer1.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[0], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[0]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer1.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[1], config))

        self.layer2 = MultiInputSequential()
        sampleDl *= 2
        self.layer2.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[1], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[1]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer2.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[2], config))

        self.layer3 = MultiInputSequential()
        sampleDl *= 2
        self.layer3.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[2], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[2]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer3.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[3], config))

        self.layer4 = MultiInputSequential()
        sampleDl *= 2
        self.layer4.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[3], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[3]))
        radius *= 2
        width *= 2
        for i in range(depth - 1):
            self.layer4.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[4], config))

    def forward(self, xyz, mask, features, end_points=None):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, 3, input_features_dim), input points features.
            end_points: a dict

        Returns:
            end_points: a dict contains all outputs
        """
        if not end_points: end_points = {}
        # res1
        features = self.conv1(features)
        features = self.la1(xyz, xyz, mask, mask, features)
        xyz, mask, features = self.btnk1(xyz, mask, features)
        end_points['res1_xyz'] = xyz
        end_points['res1_mask'] = mask
        end_points['res1_features'] = features

        # res2
        xyz, mask, features = self.layer1(xyz, mask, features)
        end_points['res2_xyz'] = xyz
        end_points['res2_mask'] = mask
        end_points['res2_features'] = features

        # res3
        xyz, mask, features = self.layer2(xyz, mask, features)
        end_points['res3_xyz'] = xyz
        end_points['res3_mask'] = mask
        end_points['res3_features'] = features

        # res4
        xyz, mask, features = self.layer3(xyz, mask, features)
        end_points['res4_xyz'] = xyz
        end_points['res4_mask'] = mask
        end_points['res4_features'] = features

        # res5
        xyz, mask, features = self.layer4(xyz, mask, features)
        end_points['res5_xyz'] = xyz
        end_points['res5_mask'] = mask
        end_points['res5_features'] = features

        return end_points
