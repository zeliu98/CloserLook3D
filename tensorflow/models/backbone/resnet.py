import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from ..local_aggregation_operators import *


def simple_block(layer_ind,
                 config,
                 inputs,
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
    """This Block performing a simple Local Aggregation Operation

    Args:
        layer_ind: which layer to perform local aggregation
        config: config file
        inputs: a dict contains all inputs
        features: float32[n0_points, in_fdim] - input features
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        float32[n_points, out_fdim]
    """

    with tf.variable_scope(scope) as sc:
        x = LocalAggregation(config,
                             query_points=inputs['points'][layer_ind],
                             support_points=inputs['points'][layer_ind],
                             neighbors_indices=inputs['neighbors'][layer_ind],
                             features=features,
                             scope='local_aggreagtion',
                             radius=radius,
                             out_fdim=out_fdim,
                             is_training=is_training,
                             init=init,
                             weight_decay=weight_decay,
                             activation_fn=activation_fn,
                             bn=bn,
                             bn_momentum=bn_momentum,
                             bn_eps=bn_eps)
        return x


def bottleneck(layer_ind,
               config,
               inputs,
               features,
               scope,
               radius,
               out_fdim,
               bottleneck_ratio,
               is_training,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """This Block performing a resnet bottleneck convolution (1conv > Local Aggregation > 1conv + shortcut)

    Args:
        layer_ind: which layer to perform local aggregation
        config: config file
        inputs: a dict contains all inputs
        features: float32[n_points, in_fdim] - input features
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        bottleneck_ratio: bottleneck_ratio
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        float32[n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('conv1'):
            x = conv1d_1x1(features,
                           out_fdim // bottleneck_ratio,
                           scope='conv1d_1x1',
                           is_training=is_training,
                           with_bias=False,
                           init=init,
                           weight_decay=weight_decay,
                           activation_fn=activation_fn,
                           bn=bn,
                           bn_momentum=bn_momentum,
                           bn_eps=bn_eps)

        with tf.variable_scope('conv2'):
            x = LocalAggregation(config,
                                 query_points=inputs['points'][layer_ind],
                                 support_points=inputs['points'][layer_ind],
                                 neighbors_indices=inputs['neighbors'][layer_ind],
                                 features=x,
                                 scope='local_aggregation',
                                 radius=radius,
                                 out_fdim=out_fdim // bottleneck_ratio,
                                 is_training=is_training,
                                 init=init,
                                 weight_decay=weight_decay,
                                 activation_fn=activation_fn,
                                 bn=bn,
                                 bn_momentum=bn_momentum,
                                 bn_eps=bn_eps)

        with tf.variable_scope('conv3'):
            x = conv1d_1x1(x,
                           out_fdim,
                           scope='conv1d_1x1',
                           is_training=is_training,
                           with_bias=False,
                           init=init,
                           weight_decay=weight_decay,
                           activation_fn=None,
                           bn=bn,
                           bn_momentum=bn_momentum,
                           bn_eps=bn_eps)

        with tf.variable_scope('shortcut'):
            if int(features.shape[1]) != out_fdim:
                shortcut = conv1d_1x1(features,
                                      out_fdim,
                                      scope='conv1d_1x1',
                                      is_training=is_training,
                                      with_bias=False,
                                      init=init,
                                      weight_decay=weight_decay,
                                      activation_fn=None,
                                      bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
            else:
                shortcut = features

        if activation_fn == 'relu':
            output = tf.nn.relu(x + shortcut)
        elif activation_fn == 'leaky_relu':
            output = tf.nn.leaky_relu(x + shortcut, alpha=0.2)
        else:
            output = x + shortcut

        return output


def strided_bottleneck(layer_ind,
                       config,
                       inputs,
                       features,
                       scope,
                       radius,
                       out_fdim,
                       bottleneck_ratio,
                       is_training,
                       init='xavier',
                       weight_decay=0,
                       activation_fn='relu',
                       bn=True,
                       bn_momentum=0.98,
                       bn_eps=1e-3):
    """This Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)

    Args:
        layer_ind: layer of support points, and layer+1 is of query points
        config: config file
        inputs: a dict contains all inputs
        features: float32[n0_points, in_fdim] - input features
        scope: tensorflow scope name
        radius: ball query radius
        out_fdim: output feature dim
        bottleneck_ratio: bottleneck_ratio
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        float32[n_points, in_fdim]
    """

    if out_fdim is None:
        out_fdim = features.shape[1]
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('conv1'):
            x = conv1d_1x1(features,
                           out_fdim // bottleneck_ratio,
                           scope='conv1d_1x1',
                           is_training=is_training,
                           with_bias=False,
                           init=init,
                           weight_decay=weight_decay,
                           activation_fn=activation_fn,
                           bn=bn,
                           bn_momentum=bn_momentum,
                           bn_eps=bn_eps)

        with tf.variable_scope('conv2'):
            x = LocalAggregation(config,
                                 query_points=inputs['points'][layer_ind + 1],
                                 support_points=inputs['points'][layer_ind],
                                 neighbors_indices=inputs['pools'][layer_ind],
                                 features=x,
                                 scope='local_aggregation',
                                 radius=radius,
                                 out_fdim=out_fdim // bottleneck_ratio,
                                 is_training=is_training,
                                 init=init,
                                 weight_decay=weight_decay,
                                 activation_fn=activation_fn,
                                 bn=bn,
                                 bn_momentum=bn_momentum,
                                 bn_eps=bn_eps)

        with tf.variable_scope('conv3'):
            x = conv1d_1x1(x,
                           out_fdim,
                           scope='conv1d_1x1',
                           is_training=is_training,
                           with_bias=False,
                           init=init,
                           weight_decay=weight_decay,
                           activation_fn=None,
                           bn=bn,
                           bn_momentum=bn_momentum,
                           bn_eps=bn_eps)

        with tf.variable_scope('shortcut'):
            # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
            shortcut = ind_max_pool(features, inputs['pools'][layer_ind], 'max_pool')
            # shortcut = ind_closest_pool(features, neighbors_indices,'closest pool')

            # Regular upsample of the features if not the same dimension
            if int(shortcut.shape[1]) != out_fdim:
                shortcut = conv1d_1x1(shortcut,
                                      out_fdim,
                                      scope='conv1d_1x1',
                                      is_training=is_training,
                                      with_bias=False,
                                      init=init,
                                      weight_decay=weight_decay,
                                      activation_fn=None,
                                      bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
        if activation_fn == 'relu':
            output = tf.nn.relu(x + shortcut)
        elif activation_fn == 'leaky_relu':
            output = tf.nn.leaky_relu(x + shortcut, alpha=0.2)
        else:
            output = x + shortcut

        return output


def resnet_backbone(config,
                    inputs,
                    features,
                    base_radius,
                    base_fdim,
                    bottleneck_ratio,
                    depth,
                    is_training,
                    init='xavier',
                    weight_decay=0,
                    activation_fn='relu',
                    bn=True,
                    bn_momentum=0.98,
                    bn_eps=1e-3):
    """Resnet Backbone

    Args:
        config: config file
        inputs: a dict contains all inputs
        features: input features
        base_radius: the first ball query radius
        base_fdim: the base feature dim
        bottleneck_ratio: bottleneck_ratio
        depth: num of bottleneck in a stage
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        A list of all stage features
    """
    with tf.variable_scope('resnet_backbone') as sc:
        fdim = base_fdim
        radius = base_radius
        layer_idx = 0
        F = []
        features = conv1d_1x1(features, fdim, 'res1_input_conv', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        features = simple_block(layer_idx, config, inputs, features, 'res1_simple_block',
                                radius=radius, out_fdim=fdim, is_training=is_training,
                                init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                bn_momentum=bn_momentum, bn_eps=bn_eps)
        for i in range(depth):
            features = bottleneck(layer_idx, config, inputs, features, f'res1_bottleneck{i}',
                                  radius=radius, out_fdim=2 * fdim, bottleneck_ratio=bottleneck_ratio,
                                  is_training=is_training,
                                  init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                  bn_momentum=bn_momentum,
                                  bn_eps=bn_eps)
        F += [features]
        layer_idx += 1
        features = strided_bottleneck(layer_idx - 1, config, inputs, features, 'res2_strided_bottleneck',
                                      radius=radius, out_fdim=4 * fdim, bottleneck_ratio=bottleneck_ratio,
                                      is_training=is_training,
                                      init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
        for i in range(depth):
            features = bottleneck(layer_idx, config, inputs, features, f'res2_bottleneck{i}',
                                  radius=2 * radius, out_fdim=4 * fdim, bottleneck_ratio=bottleneck_ratio,
                                  is_training=is_training,
                                  init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                  bn_momentum=bn_momentum,
                                  bn_eps=bn_eps)
        F += [features]
        layer_idx += 1
        features = strided_bottleneck(layer_idx - 1, config, inputs, features, 'res3_strided_bottleneck',
                                      radius=2 * radius, out_fdim=8 * fdim, bottleneck_ratio=bottleneck_ratio,
                                      is_training=is_training,
                                      init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
        for i in range(depth):
            features = bottleneck(layer_idx, config, inputs, features, f'res3_bottleneck{i}',
                                  radius=4 * radius, out_fdim=8 * fdim, bottleneck_ratio=bottleneck_ratio,
                                  is_training=is_training,
                                  init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                  bn_momentum=bn_momentum,
                                  bn_eps=bn_eps)
        F += [features]
        layer_idx += 1
        features = strided_bottleneck(layer_idx - 1, config, inputs, features, 'res4_strided_bottleneck',
                                      radius=4 * radius, out_fdim=16 * fdim, bottleneck_ratio=bottleneck_ratio,
                                      is_training=is_training,
                                      init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
        for i in range(depth):
            features = bottleneck(layer_idx, config, inputs, features, f'res4_bottleneck{i}',
                                  radius=8 * radius, out_fdim=16 * fdim, bottleneck_ratio=bottleneck_ratio,
                                  is_training=is_training,
                                  init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                  bn_momentum=bn_momentum,
                                  bn_eps=bn_eps)
        F += [features]
        layer_idx += 1
        features = strided_bottleneck(layer_idx - 1, config, inputs, features, 'res5_strided_bottleneck',
                                      radius=8 * radius, out_fdim=32 * fdim, bottleneck_ratio=bottleneck_ratio,
                                      is_training=is_training,
                                      init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                      bn_momentum=bn_momentum,
                                      bn_eps=bn_eps)
        for i in range(depth):
            features = bottleneck(layer_idx, config, inputs, features, f'res5_bottleneck{i}',
                                  radius=16 * radius, out_fdim=32 * fdim, bottleneck_ratio=bottleneck_ratio,
                                  is_training=is_training,
                                  init=init, weight_decay=weight_decay, activation_fn=activation_fn, bn=bn,
                                  bn_momentum=bn_momentum,
                                  bn_eps=bn_eps)
        F += [features]

    return F
