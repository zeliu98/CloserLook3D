import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.append(ROOT_DIR)

from ..local_aggregation_operators import *


def resnet_classification_head(config,
                               inputs,
                               features,
                               base_fdim,
                               is_training,
                               pooling='avg',
                               init='xavier',
                               weight_decay=0,
                               activation_fn='relu',
                               bn=True,
                               bn_momentum=0.98,
                               bn_eps=1e-3):
    """A head for shape classification with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        features: input features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        pooling: global pooling type, avg or max
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        prediction logits [batch_size, num_classes]
    """
    with tf.variable_scope('resnet_classification_head') as sc:
        fdim = base_fdim
        if pooling == 'avg':
            features = global_average_block(inputs, features, 'global_avg_pool')
        elif pooling == 'max':
            features = global_max_block(inputs, features, 'global_max_pool')
        else:
            raise NotImplementedError(f"{pooling} not supported in resnet_classification_head")

        features = conv1d_1x1(features, 16 * fdim, 'fc1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        features = dropout(features, keep_prob=0.5, is_training=is_training, scope='dp1')

        features = conv1d_1x1(features, 8 * fdim, 'fc2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        features = dropout(features, keep_prob=0.5, is_training=is_training, scope='dp2')

        features = conv1d_1x1(features, 4 * fdim, 'fc3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        features = dropout(features, keep_prob=0.5, is_training=is_training, scope='dp3')

        logits = conv1d_1x1(features, config.num_classes, 'logit', is_training=is_training, with_bias=True, init=init,
                            weight_decay=weight_decay, activation_fn=None, bn=False)
    return logits
