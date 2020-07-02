import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from .basic_operators import _variable_with_weight_decay
from .basic_operators import *
from .utlis import *


def PosPool(config,
            query_points,
            support_points,
            neighbors_indices,
            features,
            scope,
            radius,
            out_fdim,
            is_training,
            init='xavier',
            weight_decay=0,
            activation_fn='relu',
            bn=True,
            bn_momentum=0.98,
            bn_eps=1e-3):
    """A PosPool operator for local aggregation

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        position_embedding = config.pospool.position_embedding
        reduction = config.pospool.reduction
        output_conv = config.pospool.output_conv

        # some shapes
        n_points = tf.shape(query_points)[0]
        n0_points = tf.shape(support_points)[0]
        n_neighbors = tf.shape(neighbors_indices)[1]
        fdim = int(features.shape[1])

        # deal with input feature
        # Add a fake feature in the last row for shadow neighbors
        shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        # Get the features of each neighborhood [n_points, n_neighbors, fdim]
        neighborhood_features = tf.gather(shadow_features, neighbors_indices, axis=0)

        # deal with input point position
        shadow_points = tf.concat([support_points, tf.zeros_like(support_points[:1, :])], axis=0)
        neighbor_points = tf.gather(shadow_points, neighbors_indices, axis=0)
        center_points = tf.expand_dims(query_points, 1)
        relative_position = neighbor_points - center_points
        relative_position = (relative_position / radius)  # norm position (d/r)
        distances = tf.sqrt(tf.reduce_sum(tf.square(relative_position), axis=2, keep_dims=True))
        direction = relative_position / (distances + 1e-6)

        # get Position Embedding
        if position_embedding == 'one':
            geo_prior = tf.ones_like(distances)
            mid_fdim = 1
            shared_channels = fdim
        elif position_embedding == 'xyz':
            geo_prior = relative_position
            mid_fdim = 3
            shared_channels = fdim // 3
        elif position_embedding == 'distance':
            geo_prior = distances
            mid_fdim = 1
            shared_channels = fdim // 1
        elif position_embedding == 'exp_-d':
            distances = tf.exp(-1.0 * distances)
            geo_prior = distances
            mid_fdim = 1
            shared_channels = fdim // 1
        elif position_embedding == 'direction':
            geo_prior = direction
            mid_fdim = 1
            shared_channels = fdim // 1
        elif position_embedding == 'direction_exp_-d':
            distances = tf.exp(-1.0 * distances)
            if fdim <= 18:
                geo_prior = tf.concat([direction, distances, direction, distances, distances], axis=-1)
                mid_fdim = 9
                shared_channels = fdim // 9
            else:
                geo_prior = tf.concat([direction, distances], axis=-1)
                mid_fdim = 4
                shared_channels = fdim // 4
        elif position_embedding == 'direction_d':
            if fdim <= 18:
                geo_prior = tf.concat([direction, distances, direction, distances, distances], axis=-1)
                mid_fdim = 9
                shared_channels = fdim // 9
            else:
                geo_prior = tf.concat([direction, distances], axis=-1)
                mid_fdim = 4
                shared_channels = fdim // 4
        elif position_embedding == 'sin_cos':
            position_mat = relative_position  # [n_points, n_neighbors, 3]
            if fdim == 9:
                feat_dim = 1
                wave_length = 1000
                alpha = 100
                feat_range = tf.range(feat_dim, dtype=np.float32)  # (feat_dim, )
                dim_mat = tf.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
                position_mat = tf.expand_dims(alpha * position_mat, -1)  # [n_points, n_neighbors, 3, 1]
                div_mat = tf.div(position_mat, dim_mat)  # [n_points, n_neighbors, 3, feat_dim]
                sin_mat = tf.sin(div_mat)  # [n_points, n_neighbors, 3, feat_dim]
                cos_mat = tf.cos(div_mat)  # [n_points, n_neighbors, 3, feat_dim]
                embedding = tf.concat([sin_mat, cos_mat], -1)  # [n_points, n_neighbors, 3, 2*feat_dim]
                embedding = tf.reshape(embedding,
                                       [n_points, n_neighbors, 6])  # [n_points, n_neighbors, 6*feat_dim]
                embedding = tf.concat([embedding, relative_position], -1)  # [n_points, n_neighbors, 9]
                geo_prior = embedding  # [n_points, n_neighbors, mid_dim]
            else:
                feat_dim = fdim // 6
                wave_length = 1000
                alpha = 100
                feat_range = tf.range(feat_dim, dtype=np.float32)  # (feat_dim, )
                dim_mat = tf.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
                position_mat = tf.expand_dims(alpha * position_mat, -1)  # [n_points, n_neighbors, 3, 1]
                div_mat = tf.div(position_mat, dim_mat)  # [n_points, n_neighbors, 3, feat_dim]
                sin_mat = tf.sin(div_mat)  # [n_points, n_neighbors, 3, feat_dim]
                cos_mat = tf.cos(div_mat)  # [n_points, n_neighbors, 3, feat_dim]
                embedding = tf.concat([sin_mat, cos_mat], -1)  # [n_points, n_neighbors, 3, 2*feat_dim]
                embedding = tf.reshape(embedding, [n_points, n_neighbors, fdim])  # [n_points, n_neighbors, 6*feat_dim]
                geo_prior = embedding  # [n_points, n_neighbors, mid_dim]
            mid_fdim = fdim
            shared_channels = 1
        elif position_embedding == 'two_order':
            geo_prior_x = relative_position[:, :, :1]  # [n_points, n_neighbors, 1]
            geo_prior_y = relative_position[:, :, 1:2]  # [n_points, n_neighbors, 1]
            geo_prior_z = relative_position[:, :, 2:3]  # [n_points, n_neighbors, 1]
            geo_prior_xy = geo_prior_x * geo_prior_y
            geo_prior_xz = geo_prior_x * geo_prior_z
            geo_prior_yz = geo_prior_y * geo_prior_z
            geo_prior_xx = tf.square(geo_prior_x)
            geo_prior_yy = tf.square(geo_prior_y)
            geo_prior_zz = tf.square(geo_prior_z)

            geo_prior = tf.concat(
                [relative_position, geo_prior_xy, geo_prior_xz, geo_prior_yz, geo_prior_xx, geo_prior_yy, geo_prior_zz],
                axis=-1)  # [n_points, n_neighbors, 9]
            mid_fdim = 9
            shared_channels = fdim // 9
        elif position_embedding == 'three_order':
            # first order
            geo_prior_x = relative_position[:, :, :1]  # [n_points, n_neighbors, 1]
            geo_prior_y = relative_position[:, :, 1:2]  # [n_points, n_neighbors, 1]
            geo_prior_z = relative_position[:, :, 2:3]  # [n_points, n_neighbors, 1]
            # two order
            geo_prior_xy = geo_prior_x * geo_prior_y
            geo_prior_xz = geo_prior_x * geo_prior_z
            geo_prior_yz = geo_prior_y * geo_prior_z
            geo_prior_xx = tf.square(geo_prior_x)
            geo_prior_yy = tf.square(geo_prior_y)
            geo_prior_zz = tf.square(geo_prior_z)
            # third order
            geo_prior_xxx = tf.pow(geo_prior_x, 3)
            geo_prior_yyy = tf.pow(geo_prior_y, 3)
            geo_prior_zzz = tf.pow(geo_prior_z, 3)
            geo_prior_xxy = geo_prior_xx * geo_prior_y
            geo_prior_xxz = geo_prior_xx * geo_prior_z
            geo_prior_yyx = geo_prior_yy * geo_prior_x
            geo_prior_yyz = geo_prior_yy * geo_prior_z
            geo_prior_zzx = geo_prior_zz * geo_prior_x
            geo_prior_zzy = geo_prior_zz * geo_prior_y
            # geo_prior_xyz = geo_prior_x * geo_prior_y * geo_prior_z

            if fdim == 9:
                geo_prior = tf.concat(
                    [relative_position, geo_prior_xy, geo_prior_xz, geo_prior_yz, geo_prior_xx, geo_prior_yy,
                     geo_prior_zz],
                    axis=-1)  # [n_points, n_neighbors, 9]
                mid_fdim = 9
                shared_channels = 1
            else:
                geo_prior = tf.concat(
                    [relative_position, geo_prior_xy, geo_prior_xz, geo_prior_yz, geo_prior_xx, geo_prior_yy,
                     geo_prior_zz, geo_prior_xxx, geo_prior_yyy, geo_prior_zzz, geo_prior_xxy, geo_prior_xxz,
                     geo_prior_yyx, geo_prior_yyz, geo_prior_zzx, geo_prior_zzy],
                    axis=-1)  # [n_points, n_neighbors, 18]
                mid_fdim = 18
                shared_channels = fdim // 18
        else:
            raise NotImplementedError("position_embedding [{}] not supported in PosPool ".format(position_embedding))

        geo_prior = tf.expand_dims(geo_prior, -1)  # [n_points, n_neighbors, mid_fdim, 1]
        feature_map = tf.reshape(neighborhood_features, [n_points, n_neighbors, mid_fdim, shared_channels])
        aggregation_feature = tf.multiply(geo_prior, feature_map)
        aggregation_feature = tf.reshape(aggregation_feature, [n_points, -1, fdim])  # [n_points,n_neighbors, fdim]

        if reduction == 'sum':
            aggregation_feature = tf.reduce_sum(aggregation_feature, 1)  # [n_points, fdim]
        elif reduction == 'mean' or reduction == 'avg':
            aggregation_feature = tf.reduce_sum(aggregation_feature, 1)  # [n_points, fdim]
            padding_num = tf.reduce_max(neighbors_indices)
            neighbors_n = tf.where(tf.less(neighbors_indices, padding_num), tf.ones_like(neighbors_indices),
                                   tf.zeros_like(neighbors_indices))
            neighbors_n = tf.cast(neighbors_n, tf.float32)
            neighbors_n = tf.reduce_sum(neighbors_n, -1, keep_dims=True) + 1e-5  # [n_points, 1]
            aggregation_feature = aggregation_feature / neighbors_n
        elif reduction == 'max':
            # mask padding
            batch_mask = tf.zeros((n0_points, fdim), dtype=tf.float32)  # [n0_points]
            batch_mask = tf.concat([batch_mask, -65535 * tf.ones_like(batch_mask[:1])], axis=0)  # [n0_points+1, fdim]
            batch_mask = tf.gather(batch_mask, neighbors_indices, axis=0)  # [n_points, n_neighbors, fdim]
            aggregation_feature = aggregation_feature + batch_mask
            aggregation_feature = tf.reduce_max(aggregation_feature, 1)  # [n_points, fdim]
        else:
            raise NotImplementedError("Reduction {} not supported in PosPool".format(reduction))

        if bn:
            aggregation_feature = batch_norm(aggregation_feature, is_training=is_training, scope='pool_bn',
                                             bn_decay=bn_momentum, epsilon=bn_eps)
        if activation_fn == 'relu':
            aggregation_feature = tf.nn.relu(aggregation_feature)
        elif activation_fn == 'leaky_relu':
            aggregation_feature = tf.nn.leaky_relu(aggregation_feature, alpha=0.2)

        # Output
        if fdim != out_fdim or output_conv:
            output_features = conv1d_1x1(aggregation_feature, out_fdim, scope='output_conv',
                                         is_training=is_training,
                                         init=init, weight_decay=weight_decay,
                                         activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                                         bn_eps=bn_eps)  # [n_points, out_fdim]
        else:
            output_features = aggregation_feature
        return output_features


def Identity(config,
             query_points,
             support_points,
             neighbors_indices,
             features,
             scope,
             radius,
             out_fdim,
             is_training,
             init='xavier',
             weight_decay=0,
             activation_fn='relu',
             bn=True,
             bn_momentum=0.98,
             bn_eps=1e-3):
    """An Identity operator to replace local aggregation

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution


    Returns:
        [n_points, out_fdim]
    """

    with tf.variable_scope(scope) as sc:
        n_points = tf.shape(query_points)[0]
        fdim = int(features.shape[1])

        shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        # Get the features of query points [n_points, 1, fdim]
        center_features = tf.gather(shadow_features, neighbors_indices[:, :1], axis=0)
        center_features = tf.reshape(center_features, [n_points, fdim])  # [n_points, fdim]
        if fdim != out_fdim:
            output_features = conv1d_1x1(center_features, out_fdim, scope='output_conv',
                                         is_training=is_training,
                                         init=init, weight_decay=weight_decay,
                                         activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                                         bn_eps=bn_eps)  # [n_points, out_fdim]
        else:
            if bn:
                center_features = batch_norm(center_features, is_training=is_training, scope='pool_bn',
                                             bn_decay=bn_momentum, epsilon=bn_eps)
            if activation_fn == 'relu':
                center_features = tf.nn.relu(center_features)
            elif activation_fn == 'leaky_relu':
                center_features = tf.nn.leaky_relu(center_features, alpha=0.2)
            output_features = center_features

        return output_features


def AdaptiveWeight(config,
                   query_points,
                   support_points,
                   neighbors_indices,
                   features,
                   scope,
                   radius,
                   out_fdim,
                   is_training,
                   init='xavier',
                   weight_decay=0,
                   activation_fn='relu',
                   bn=True,
                   bn_momentum=0.98,
                   bn_eps=1e-3):
    """An Adaptive Weight operator for local aggregation

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        # some configs
        local_input_feature = config.adaptive_weight.local_input_feature
        reduction = config.adaptive_weight.reduction
        shared_channels = config.adaptive_weight.shared_channels
        fc_num = config.adaptive_weight.fc_num
        weight_softmax = config.adaptive_weight.weight_softmax
        output_conv = config.adaptive_weight.output_conv
        fdim = int(features.shape[1])
        if shared_channels > fdim:
            shared_channels = fdim
        mid_fdim = fdim // shared_channels
        # some shapes
        n_points = tf.shape(query_points)[0]
        n0_points = tf.shape(support_points)[0]
        n_neighbors = tf.shape(neighbors_indices)[1]

        # Deal with input features
        # Add a fake feature in the last row for shadow neighbors
        shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        # Get the features of each neighborhood [n_points, n_neighbors, fdim]
        neighborhood_features = tf.gather(shadow_features, neighbors_indices, axis=0)
        # Get the features of query points [n_points, 1, fdim]
        center_features = tf.gather(shadow_features, neighbors_indices[:, :1], axis=0)

        # Deal with input point position
        relative_features = neighborhood_features - center_features
        shadow_points = tf.concat([support_points, tf.zeros_like(support_points[:1, :])], axis=0)
        neighbor_points = tf.gather(shadow_points, neighbors_indices, axis=0)
        center_points = tf.expand_dims(query_points, 1)
        relative_position = neighbor_points - center_points
        relative_position = (relative_position / radius)  # norm position (d/r)
        distances = tf.sqrt(tf.reduce_sum(tf.square(relative_position), axis=2, keep_dims=True))

        # get Conv Weight
        if local_input_feature == 'dp':
            conv_weight = relative_position
        elif local_input_feature == 'df':
            conv_weight = relative_features
        elif local_input_feature == 'dp_df':
            conv_weight = tf.concat([relative_position, relative_features], axis=-1)
        elif local_input_feature == 'fj':
            conv_weight = neighborhood_features
        elif local_input_feature == 'dp_fj':
            conv_weight = tf.concat([relative_position, neighborhood_features], axis=-1)
        elif local_input_feature == 'fi_df':
            center_features = tf.tile(center_features, [1, n_neighbors, 1])
            conv_weight = tf.concat([center_features, relative_features], axis=-1)
        elif local_input_feature == 'dp_fi_df':
            center_features = tf.tile(center_features, [1, n_neighbors, 1])
            conv_weight = tf.concat([relative_position, center_features, relative_features], axis=-1)
        elif local_input_feature == 'rscnn':
            center_points = tf.tile(center_points, [1, n_neighbors, 1])
            conv_weight = tf.concat([distances, relative_position, center_points, neighbor_points],
                                    axis=-1)  # [n_points, n_neighbors, 10]
        elif local_input_feature == 'gac':
            features = conv1d_1x1(features, mid_fdim, scope='GAC_conv1',
                                  init=init, weight_decay=weight_decay,
                                  activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum, bn_eps=bn_eps,
                                  is_training=is_training)  # [n_points, mid_fdim]
            shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
            neighborhood_features = tf.gather(shadow_features, neighbors_indices, axis=0)
            center_features = tf.gather(shadow_features, neighbors_indices[:, :1], axis=0)
            relative_features = neighborhood_features - center_features
            conv_weight = tf.concat([relative_position, relative_features], axis=-1)
        else:
            raise NotImplementedError(
                "Local input feature {} not supported in AdaptiveWeight".format(local_input_feature))
        for i in range(fc_num - 1):
            conv_weight = batch_conv1d_1x1(conv_weight, mid_fdim, scope='fc_{}'.format(i),
                                           is_training=is_training,
                                           with_bias=True,
                                           init='fan_in', weight_decay=0,
                                           activation_fn=activation_fn,
                                           bn=False)  # [n_points, n_neighbors, mid_fdim]
        conv_weight = batch_conv1d_1x1(conv_weight, mid_fdim, scope='fc_{}'.format(fc_num),
                                       is_training=is_training,
                                       with_bias=True,
                                       init='fan_in', weight_decay=0,
                                       activation_fn=None, bn=False)  # [n_points, n_neighbors, mid_fdim]
        # Softmax
        if weight_softmax == 'dense':
            batch_mask = tf.ones((n0_points), dtype=tf.bool)  # [n0_points]
            batch_mask = tf.concat([batch_mask, tf.zeros_like(batch_mask[:1], dtype=tf.bool)],
                                   axis=0)  # [n0_points+1]
            batch_mask = tf.gather(batch_mask, neighbors_indices, axis=0)  # [n_points, n_neighbors]
            conv_weight = dense_masked_softmax(conv_weight, batch_mask, mid_fdim)  # [n_points, n_neighbors, mid_fdim]
        elif weight_softmax == 'sparse':
            batch_mask = tf.ones((n0_points, mid_fdim), dtype=tf.bool)  # [n0_points,mid_fdim]
            batch_mask = tf.concat([batch_mask, tf.zeros_like(batch_mask[:1, :], dtype=tf.bool)],
                                   axis=0)  # [n0_points+1,mid_fdim]
            batch_mask = tf.gather(batch_mask, neighbors_indices, axis=0)  # [n_points, n_neighbors,mid_fdim]
            conv_weight = tf.transpose(conv_weight, [0, 2, 1])  # [n_points, mid_fdim, n_neighbors]
            batch_mask = tf.transpose(batch_mask, [0, 2, 1])  # [n_points, mid_fdim, n_neighbors]
            conv_weight = sparse_masked_softmax(conv_weight, batch_mask)  # [n_points,mid_fdim, n_neighbors]
            conv_weight = tf.transpose(conv_weight, [0, 2, 1])  # [n_points, n_neighbors, mid_fdim]
        elif weight_softmax == 'unmask':
            conv_weight = tf.nn.softmax(conv_weight, 1)  # [n_points, n_neighbors, mid_fdim]

        # Transformation
        conv_weight = tf.expand_dims(conv_weight, -1)  # [n_points, n_neighbors, mid_fdim, 1]
        feature_map = tf.reshape(neighborhood_features, [n_points, n_neighbors, mid_fdim, shared_channels])
        aggregation_feature = tf.multiply(conv_weight, feature_map)
        aggregation_feature = tf.reshape(aggregation_feature, [n_points, -1, fdim])  # [n_points,n_neighbors, fdim]

        # Reduction
        if reduction == 'sum':
            aggregation_feature = tf.reduce_sum(aggregation_feature, 1)  # [n_points, fdim]
        elif reduction == 'mean' or reduction == 'avg':
            aggregation_feature = tf.reduce_sum(aggregation_feature, 1)  # [n_points, fdim]
            padding_num = tf.reduce_max(neighbors_indices)
            neighbors_n = tf.where(tf.less(neighbors_indices, padding_num), tf.ones_like(neighbors_indices),
                                   tf.zeros_like(neighbors_indices))
            neighbors_n = tf.cast(neighbors_n, tf.float32)
            neighbors_n = tf.reduce_sum(neighbors_n, -1, keep_dims=True) + 1e-5  # [n_points, 1]
            aggregation_feature = aggregation_feature / neighbors_n
        elif reduction == 'max':
            # mask padding
            batch_mask = tf.zeros((n0_points, fdim), dtype=tf.float32)  # [n0_points]
            batch_mask = tf.concat([batch_mask, -65535 * tf.ones_like(batch_mask[:1])], axis=0)  # [n0_points+1, fdim]
            batch_mask = tf.gather(batch_mask, neighbors_indices, axis=0)  # [n_points, n_neighbors, fdim]
            aggregation_feature = aggregation_feature + batch_mask
            aggregation_feature = tf.reduce_max(aggregation_feature, 1)  # [n_points, fdim]
        else:
            raise NotImplementedError("Reduction {} not supported in PosPool".format(reduction))

        if bn:
            aggregation_feature = batch_norm(aggregation_feature, is_training=is_training, scope='pool_bn',
                                             bn_decay=bn_momentum, epsilon=bn_eps)
        if activation_fn == 'relu':
            aggregation_feature = tf.nn.relu(aggregation_feature)
        elif activation_fn == 'leaky_relu':
            aggregation_feature = tf.nn.leaky_relu(aggregation_feature, alpha=0.2)

        # Output
        if fdim != out_fdim or output_conv:
            output_features = conv1d_1x1(aggregation_feature, out_fdim,
                                         is_training=is_training,
                                         scope='output_conv',
                                         init=init, weight_decay=weight_decay,
                                         activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                                         bn_eps=bn_eps)  # [n_points, out_fdim]
        else:
            output_features = aggregation_feature
        return output_features


def PointWiseMLP(config,
                 query_points,
                 support_points,
                 neighbors_indices,
                 features,
                 scope,
                 radius,
                 out_fdim,
                 is_training,
                 init='xavier',
                 weight_decay=0,
                 activation_fn='relu',
                 bn=True,
                 bn_momentum=0.98,
                 bn_eps=1e-3):
    """An Point-wise MLP operator for local aggregation

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        # some configs
        local_input_feature = config.pointwisemlp.local_input_feature
        fc_num = config.pointwisemlp.fc_num
        reduction = config.pointwisemlp.reduction
        fdim = int(features.shape[1])

        # some shapes
        n_points = tf.shape(query_points)[0]
        n0_points = tf.shape(support_points)[0]
        n_neighbors = tf.shape(neighbors_indices)[1]

        # Deal with input feature
        # Add a fake feature in the last row for shadow neighbors
        shadow_features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        # Get the features of each neighborhood [n_points, n_neighbors, fdim]
        neighborhood_features = tf.gather(shadow_features, neighbors_indices, axis=0)
        # Get the features of query points [n_points, 1, fdim]
        center_features = tf.gather(shadow_features, neighbors_indices[:, :1], axis=0)
        relative_features = neighborhood_features - center_features

        # Deal with input point position
        shadow_points = tf.concat([support_points, tf.zeros_like(support_points[:1, :])], axis=0)
        neighbor_points = tf.gather(shadow_points, neighbors_indices, axis=0)
        center_points = tf.expand_dims(query_points, 1)
        relative_position = neighbor_points - center_points
        relative_position = (relative_position / radius)  # norm position (d/r)

        # mask padding
        batch_mask = tf.ones((n0_points), dtype=tf.float32)  # [n0_points]
        batch_mask = tf.concat([batch_mask, tf.zeros_like(batch_mask[:1])], axis=0)  # [n0_points+1]
        batch_mask = tf.gather(batch_mask, neighbors_indices, axis=0)  # [n_points, n_neighbors]
        batch_mask = tf.expand_dims(batch_mask, -1)  # [n_points, n_neighbors,1]
        batch_mask = tf.tile(batch_mask, [1, 1, out_fdim])  # [n_points, n_neighbors, out_fdim]

        if local_input_feature == 'dp_fj':
            set_features = tf.concat([relative_position, neighborhood_features], axis=-1)
        elif local_input_feature == 'fi_df':
            center_features = tf.tile(center_features, [1, n_neighbors, 1])
            set_features = tf.concat([center_features, relative_features], axis=-1)
        elif local_input_feature == 'dp_fi_df':
            center_features = tf.tile(center_features, [1, n_neighbors, 1])
            set_features = tf.concat([relative_position, center_features, relative_features], axis=-1)
        elif local_input_feature == 'dp_fi_df_fj':
            center_features = tf.tile(center_features, [1, n_neighbors, 1])
            set_features = tf.concat([relative_position, center_features, relative_features, neighborhood_features],
                                     axis=-1)
        else:
            raise NotImplementedError(
                "local_input_feature {} not supported in Point-wise MLP".format(local_input_feature))

        mfdim = max(fdim // 2, 9)
        for i in range(fc_num - 1):
            set_features = batch_conv1d_1x1(set_features, mfdim, scope='fc_{}'.format(i),
                                            is_training=is_training,
                                            init=init, weight_decay=weight_decay,
                                            activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                                            bn_eps=bn_eps)  # [n_points, n_neighbors, fdim]
        set_features = batch_conv1d_1x1(set_features, out_fdim, scope='fc_{}'.format(fc_num),
                                        is_training=is_training,
                                        init=init, weight_decay=weight_decay,
                                        activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                                        bn_eps=bn_eps)  # [n_points, n_neighbors, out_fdim]
        set_features = set_features * batch_mask

        if reduction == 'max':
            output_feature = tf.reduce_max(set_features, 1)  # [n_points, fdim]
        elif reduction == 'sum':
            output_feature = tf.reduce_sum(set_features, 1)
        elif reduction == 'mean':
            aggregation_feature = tf.reduce_sum(set_features, 1)  # [n_points, fdim]
            padding_num = tf.reduce_max(neighbors_indices)
            neighbors_n = tf.where(tf.less(neighbors_indices, padding_num), tf.ones_like(neighbors_indices),
                                   tf.zeros_like(neighbors_indices))
            neighbors_n = tf.cast(neighbors_n, tf.float32)
            neighbors_n = tf.reduce_sum(neighbors_n, -1, keep_dims=True) + 1e-5  # [n_points, 1]
            output_feature = aggregation_feature / neighbors_n
        else:
            raise NotImplementedError("Reduction {} not supported in Point-wise MLP.".format(reduction))
        return output_feature


def PseudoGrid(config,
               query_points,
               support_points,
               neighbors_indices,
               features,
               scope,
               radius,
               out_fdim,
               is_training,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """A PseudoGrid (KPConv) operator for local aggregation

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        fixed_kernel_points = config.pseudo_grid.fixed_kernel_points
        KP_influence = config.pseudo_grid.KP_influence
        KP_extent = config.pseudo_grid.KP_extent
        num_kernel_points = config.pseudo_grid.num_kernel_points
        convolution_mode = config.pseudo_grid.convolution_mode
        density_parameter = config.density_parameter
        output_conv = config.pseudo_grid.output_conv

        extent = KP_extent * radius / density_parameter
        K_radius = 1.5 * extent
        fdim = int(features.shape[1])

        # create kernel points
        K_points_numpy = create_kernel_points(sc,
                                              K_radius,
                                              num_kernel_points,
                                              num_kernels=1,
                                              dimension=3,
                                              fixed=fixed_kernel_points)

        K_points_numpy = K_points_numpy.reshape((num_kernel_points, 3))
        K_points = tf.constant(K_points_numpy.astype(np.float32), name='kernel_points', dtype=tf.float32)

        # Get distances
        shadow_point = tf.ones_like(support_points[:1, :]) * 1e6
        support_points = tf.concat([support_points, shadow_point], axis=0)
        neighbors = tf.gather(support_points, neighbors_indices, axis=0)
        neighbors = neighbors - tf.expand_dims(query_points, 1)
        neighbors = tf.expand_dims(neighbors, 2)
        neighbors = tf.tile(neighbors, [1, 1, num_kernel_points, 1])
        differences = neighbors - K_points
        sq_distances = tf.reduce_sum(tf.square(differences), axis=3)

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = tf.ones_like(sq_distances)
            all_weights = tf.transpose(all_weights, [0, 2, 1])
        elif KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / extent, 0.0)
            all_weights = tf.transpose(all_weights, [0, 2, 1])
        elif KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = tf.transpose(all_weights, [0, 2, 1])
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        if convolution_mode == 'closest':
            # In case of closest mode, only the closest KP can influence each point
            neighbors_1nn = tf.argmin(sq_distances, axis=2, output_type=tf.int32)
            all_weights *= tf.one_hot(neighbors_1nn, num_kernel_points, axis=1, dtype=tf.float32)
        elif convolution_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # get features for each kernel point
        features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)
        neighborhood_features = tf.gather(features, neighbors_indices, axis=0)
        weighted_features = tf.matmul(all_weights, neighborhood_features)  # [n_points, n_kpoints, in_fdim]

        # Get weight
        kernel_weights = _variable_with_weight_decay('weights',
                                                     shape=[num_kernel_points, fdim],
                                                     init=init,
                                                     wd=weight_decay)
        kernel_weights = tf.expand_dims(kernel_weights, 0)  # [1, n_kpoints, in_fdim]
        kernel_outputs = tf.multiply(kernel_weights, weighted_features)  # [n_points, n_kpoints, in_fdim]

        # Convolution sum to get [n_points, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=1)

        if bn:
            output_features = batch_norm(output_features, is_training, scope='bn', bn_decay=bn_momentum,
                                         epsilon=bn_eps)
        if activation_fn == 'relu':
            output_features = tf.nn.relu(output_features)
        elif activation_fn == 'leaky_relu':
            output_features = tf.nn.leaky_relu(output_features, alpha=0.2)

        # Output Conv
        if fdim != out_fdim or output_conv:
            output_features = conv1d_1x1(output_features, out_fdim, scope='output_conv',
                                         is_training=is_training,
                                         init=init, weight_decay=weight_decay,
                                         activation_fn=activation_fn, bn=bn)  # [n_points, out_fdim]

        return output_features


def LocalAggregation(config,
                     query_points,
                     support_points,
                     neighbors_indices,
                     features,
                     scope,
                     radius,
                     out_fdim,
                     is_training,
                     init='xavier',
                     weight_decay=0,
                     activation_fn='relu',
                     bn=True,
                     bn_momentum=0.98,
                     bn_eps=1e-3):
    """Local aggregation operator wrapper

    Args:
        config: config file
        query_points: float32[n_points, 3] - input query points (center of neighborhoods)
        support_points: float32[n0_points, 3] - input support points (from which neighbors are taken)
        neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
        features: float32[n0_points, in_fdim] - input features of support points
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        [n_points, out_fdim]
    """

    if config.local_aggreagtion == 'pospool':
        return PosPool(config,
                       query_points,
                       support_points,
                       neighbors_indices,
                       features,
                       scope,
                       radius,
                       out_fdim,
                       is_training,
                       init=init,
                       weight_decay=weight_decay,
                       activation_fn=activation_fn,
                       bn=bn,
                       bn_momentum=bn_momentum,
                       bn_eps=bn_eps)
    elif config.local_aggreagtion == 'adaptive_weight':
        return AdaptiveWeight(config,
                              query_points,
                              support_points,
                              neighbors_indices,
                              features,
                              scope,
                              radius,
                              out_fdim,
                              is_training,
                              init=init,
                              weight_decay=weight_decay,
                              activation_fn=activation_fn,
                              bn=bn,
                              bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
    elif config.local_aggreagtion == 'pointwisemlp':
        return PointWiseMLP(config,
                            query_points,
                            support_points,
                            neighbors_indices,
                            features,
                            scope,
                            radius,
                            out_fdim,
                            is_training,
                            init=init,
                            weight_decay=weight_decay,
                            activation_fn=activation_fn,
                            bn=bn,
                            bn_momentum=bn_momentum,
                            bn_eps=bn_eps)
    elif config.local_aggreagtion == 'pseudo_grid':
        return PseudoGrid(config,
                          query_points,
                          support_points,
                          neighbors_indices,
                          features,
                          scope,
                          radius,
                          out_fdim,
                          is_training,
                          init=init,
                          weight_decay=weight_decay,
                          activation_fn=activation_fn,
                          bn=bn,
                          bn_momentum=bn_momentum,
                          bn_eps=bn_eps)
    elif config.local_aggreagtion == 'identity':
        return Identity(config,
                        query_points,
                        support_points,
                        neighbors_indices,
                        features,
                        scope,
                        radius,
                        out_fdim,
                        is_training,
                        init=init,
                        weight_decay=weight_decay,
                        activation_fn=activation_fn,
                        bn=bn,
                        bn_momentum=bn_momentum,
                        bn_eps=bn_eps)
    else:
        raise NotImplementedError("Local Aggregation {} not supported.".format(config.local_aggreagtion))
