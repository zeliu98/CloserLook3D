import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

try:
    import pt_custom_ops._ext as _ext
except ImportError:
    raise ImportError(
        "Could not import _ext module.\n"
        "Please see the setup instructions in the README: "
        "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
    )


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class MaskedOrderedBallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, query_xyz, support_xyz, query_mask, support_mask):
        inds, inds_mask = _ext.masked_ordered_ball_query(query_xyz, support_xyz, query_mask,
                                                         support_mask, radius, nsample)
        ctx.mark_non_differentiable(inds, inds_mask)
        return inds, inds_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None


masked_ordered_ball_query = MaskedOrderedBallQuery.apply


class MaskedNearestQuery(Function):
    @staticmethod
    def forward(ctx, query_xyz, support_xyz, query_mask, support_mask):
        inds, inds_mask = _ext.masked_nearest_query(query_xyz, support_xyz, query_mask, support_mask)
        ctx.mark_non_differentiable(inds, inds_mask)
        return inds, inds_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


masked_nearest_query = MaskedNearestQuery.apply


class MaskedGridSubsampling(Function):
    @staticmethod
    def forward(ctx, xyz, mask, npoint, sampleDl):
        sub_xyz, sub_mask = _ext.masked_grid_subsampling(xyz, mask, npoint, sampleDl)  # B N 3

        ctx.mark_non_differentiable(sub_xyz, sub_mask)
        return sub_xyz, sub_mask

    @staticmethod
    def backward(xyz, a=None):
        return None, None, None, None


masked_grid_subsampling = MaskedGridSubsampling.apply


class MaskedQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False):
        super(MaskedQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, features=None):
        idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  query_mask, support_mask)

        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz, idx_mask
        else:
            return new_features, idx_mask


class MaskedNearestQueryAndGroup(nn.Module):
    def __init__(self, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False):
        super(MaskedNearestQueryAndGroup, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, features=None):
        idx, idx_mask = masked_nearest_query(query_xyz, support_xyz, query_mask, support_mask)

        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, 1)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, 1)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz, idx_mask
        else:
            return new_features, idx_mask


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, nsample, sampleDl):
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampleDl = sampleDl
        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True)

    def forward(self, xyz, mask, features):
        # sub sample
        sub_xyz, sub_mask = masked_grid_subsampling(xyz, mask, self.npoint, self.sampleDl)
        sub_xyz = sub_xyz.contiguous()
        sub_mask = sub_mask.contiguous()

        # masked ordered ball query
        neighborhood_features, grouped_xyz, idx_mask = self.grouper(sub_xyz, xyz, sub_mask, mask,
                                                                    features)  # (B, C, npoint, nsample)

        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # (B, C, npoint, 1)
        sub_features = torch.squeeze(sub_features, -1)  # (B, C, npoint)
        return sub_xyz, sub_mask, sub_features


class MaskedUpsample(nn.Module):
    def __init__(self, radius, nsample, mode='nearest'):
        super(MaskedUpsample, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mode = mode
        if mode == 'nearest':
            self.grouper = MaskedNearestQueryAndGroup(use_xyz=False, ret_grouped_xyz=True)
        else:
            self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True)

    def forward(self, up_xyz, xyz, up_mask, mask, features):
        # masked ordered ball query
        neighborhood_features, grouped_xyz, idx_mask = self.grouper(up_xyz, xyz, up_mask, mask,
                                                                    features)  # (B, C, nsample)
        if self.mode == 'nearest':
            up_feature = neighborhood_features[..., 0].contiguous()
        elif self.mode == 'max':
            up_feature = F.max_pool2d(neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]])
            up_feature = torch.squeeze(up_feature, -1)
        else:
            raise NotImplementedError(f"mode:{self.mode} not supported in MaskedUpsample")
        return up_feature
