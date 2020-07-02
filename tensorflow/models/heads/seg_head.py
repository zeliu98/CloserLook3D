import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from ..local_aggregation_operators import *


def nearest_upsample_block(layer_ind, inputs, features, scope):
    """
    This Block performing an upsampling by nearest interpolation
    Args:
        layer_ind: Upsampled to which layer
        inputs: a dict contains all inputs
        features: x = [n1, d]
        scope: name scope

    Returns:
        x = [n2, d]
    """

    with tf.variable_scope(scope) as sc:
        upsampled_features = ind_closest_pool(features, inputs['upsamples'][layer_ind], 'nearest_upsample')
        return upsampled_features


def resnet_multi_part_segmentation_head(config,
                                        inputs,
                                        F,
                                        base_fdim,
                                        is_training,
                                        init='xavier',
                                        weight_decay=0,
                                        activation_fn='relu',
                                        bn=True,
                                        bn_momentum=0.98,
                                        bn_eps=1e-3):
    """A head for multi-shape part segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        logits for all shapes with all parts  [num_classes, num_points, num_parts_i]
    """
    with tf.variable_scope('resnet_multi_part_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        shape_heads = []
        for i_shape in range(config.num_classes):
            head = features
            head = conv1d_1x1(head, fdim, f'shape{i_shape}_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

            head = conv1d_1x1(head, config.num_parts[i_shape], f'shape{i_shape}_pred', is_training=is_training,
                              with_bias=True, init=init,
                              weight_decay=weight_decay, activation_fn=None, bn=False)
            shape_heads.append(head)

        shape_label = inputs['super_labels']
        logits_with_point_label = [()] * config.num_classes
        for i_shape in range(config.num_classes):
            i_shape_inds = tf.where(tf.equal(shape_label, i_shape))
            logits_i = tf.gather_nd(shape_heads[i_shape], i_shape_inds)
            point_labels_i = tf.gather_nd(inputs['point_labels'], i_shape_inds)
            logits_with_point_label[i_shape] = (logits_i, point_labels_i)
        logits_all_shapes = shape_heads

    return logits_with_point_label, logits_all_shapes


def resnet_scene_segmentation_head(config,
                                   inputs,
                                   F,
                                   base_fdim,
                                   is_training,
                                   init='xavier',
                                   weight_decay=0,
                                   activation_fn='relu',
                                   bn=True,
                                   bn_momentum=0.98,
                                   bn_eps=1e-3):
    """A head for scene segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        prediction logits [num_points, num_classes]
    """
    with tf.variable_scope('resnet_scene_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = conv1d_1x1(features, fdim, 'segmentation_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        logits = conv1d_1x1(features, config.num_classes, 'segmentation_pred', is_training=is_training, with_bias=True,
                            init=init, weight_decay=weight_decay, activation_fn=None, bn=False)

    return logits
