import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))


def dense_masked_softmax(logits, mask, T):
    """ Masked softmax over dim 1, mask broadcasts over dim 2, using normal softmax

    Args:
        logits: (N, L, T)
        mask: (N, L)
        T: number of dim 2
    Returns:
        probabilities (N, L, T)
    """

    v = T
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(np.array([[65535]], dtype=np.float32), dtype=tf.float32)
    infs = tf.tile(inf, [tf.shape(indices)[0], v])
    infmask = tf.scatter_nd(
        indices=indices,
        updates=infs,
        shape=tf.shape(logits))
    _p = tf.nn.softmax(logits - infmask, axis=1)
    return _p


def sparse_masked_softmax(logits, mask):
    """Masked softmax over dim -1 using sparse softmax

    Args:
        logits: (N, L, T)
        mask: (N, L, T)

    Returns:
        probabilities (N, L, T)
    """

    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)
    return result


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd, stddev=1e-3, init='xavier'):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float.
      stddev: standard deviation of a truncated Gaussian
      init: weight initializer type

    Returns:
      Variable Tensor
    """
    if init == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    elif init == 'msra':
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif init == 'fan_in':
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd > 0:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        tf.add_to_collection('weight_losses', weight_decay)
    return var


def batch_norm(inputs, is_training, scope, bn_decay=0.99, epsilon=0.001):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    return tf.layers.batch_normalization(inputs, axis=-1,
                                         momentum=bn_decay,
                                         epsilon=epsilon,
                                         training=is_training,
                                         trainable=True,
                                         name=scope,
                                         fused=False)


def ind_max_pool(x, inds, scope):
    """ This tensorflow operation compute a maxpooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
        scope: name scope

    Returns:
        [n2, d] pooled features matrix
    """
    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.reduce_min(x, axis=0, keep_dims=True)], axis=0)
        # Get features for each pooling cell [n2, max_num, d]
        pool_features = tf.gather(x, inds, axis=0)
        # Pool the maximum
        return tf.reduce_max(pool_features, axis=1)


def ind_closest_pool(x, inds, scope):
    """This tensorflow operation compute a pooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
        scope:

    Returns:
        [n2, d] pooled features matrix
    """

    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)
        # Get features for each pooling cell [n2, d]
        pool_features = tf.gather(x, inds[:, 0], axis=0)
        return pool_features


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints
    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs


def conv1d_1x1(features,
               out_fdim,
               scope,
               is_training,
               with_bias=False,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """A simple 1x1 1D convolution

    Args:
        features: Input features, float32[n_points, in_fdim]
        out_fdim: Output features dim
        scope: name scope
        is_training: True indicates training phase
        with_bias: If True, adds a learnable bias to the output
        init: Weight initializer
        weight_decay: If > 0 , add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution


    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.matmul(features, w) + biases
        else:
            x = tf.matmul(features, w)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x


def batch_conv1d_1x1(features,
                     out_fdim,
                     scope,
                     is_training,
                     with_bias=False,
                     init='xavier',
                     weight_decay=0,
                     activation_fn='relu',
                     bn=True,
                     bn_momentum=0.98,
                     bn_eps=1e-3):
    """A simple 1x1 1D convolution for batch inputs

        Args:
            features: Input features, float32[b, n_points, in_fdim]
            out_fdim: Output features dim
            scope: name scope
            is_training: True indicates training phase
            with_bias: If True, adds a learnable bias to the output
            init: Weight initializer
            weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
            activation_fn: Activation function
            bn: If True, add batch norm after convolution


        Returns:
            [b, n_points, out_fdim]
        """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.tensordot(features, w, 1) + biases
        else:
            x = tf.tensordot(features, w, 1)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x


def global_average_block(inputs, features, scope):
    """This Block performing a global average pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]

    """
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]
        # Add a last zero features for shadow batch inds
        features = tf.concat([features, tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)
        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)
        # Average features in each batch
        batch_features = tf.reduce_sum(batch_features, axis=1)
        # batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] >= 0, tf.float32), axis=1, keep_dims=True)
        batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] < N, tf.float32), axis=1, keep_dims=True)
        features = batch_features / batch_num
    return features


def global_max_block(inputs, features, scope):
    """This Block performing a global max pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]
    """

    # Average pooling to aggregate feature in the end
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]

        # Add a last zero features for shadow batch inds
        features = tf.concat([features, -256.0 + tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)

        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)

        # Average features in each batch
        batch_features = tf.reduce_max(batch_features, axis=1)

        features = batch_features

    return features
