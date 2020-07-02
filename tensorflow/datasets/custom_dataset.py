import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OPS_DIR = os.path.join(ROOT_DIR, 'ops')
sys.path.append(ROOT_DIR)
sys.path.append(OPS_DIR)

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

tf_batch_neighbors_module = tf.load_op_library(os.path.join(OPS_DIR, 'tf_custom_ops/tf_batch_neighbors.so'))
tf_batch_subsampling_module = tf.load_op_library(os.path.join(OPS_DIR, 'tf_custom_ops/tf_batch_subsampling.so'))


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """CPP wrapper for a grid subsampling (method = barycenter for points and features
    Args:
        points: (N, 3) matrix of input points
        features: optional (N, d) matrix of features (floating number)
        labels: optional (N,) matrix of integer labels
        sampleDl: parameter defining the size of grid voxels
        verbose: 1 to display

    Returns:
        subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


def tf_batch_subsampling(points, batches_len, sampleDl):
    return tf_batch_subsampling_module.batch_grid_subsampling(points, batches_len, sampleDl)


def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
    return tf_batch_neighbors_module.batch_ordered_neighbors(queries, supports, q_batches, s_batches, radius)


class CustomDataset(object):
    def __init__(self):
        self.neighborhood_limits = None
        self.augment_scale_anisotropic = None
        self.augment_symmetries = None
        self.augment_rotation = None
        self.augment_scale_min = None
        self.augment_scale_max = None
        self.augment_noise = None

        self.label_to_names = {}

    def init_labels(self):
        """
        Initiate all label parameters given the label_to_names dict
        """
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def tf_augment_input(self, stacked_points, batch_inds):
        """
        Augment inputs with rotation, scale and noise
        """
        # Parameter
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########
        if self.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))
        elif self.augment_rotation == 'vertical':
            # Choose a random angle for each element
            theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)
            # Apply rotations
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
        elif self.augment_rotation == 'arbitrarily':
            cs0 = tf.zeros((num_batches,))
            cs1 = tf.ones((num_batches,))
            # x rotation
            thetax = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cx, sx = tf.cos(thetax), tf.sin(thetax)
            Rx = tf.stack([cs1, cs0, cs0, cs0, cx, -sx, cs0, sx, cx], axis=1)
            Rx = tf.reshape(Rx, (-1, 3, 3))
            # y rotation
            thetay = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cy, sy = tf.cos(thetay), tf.sin(thetay)
            Ry = tf.stack([cy, cs0, -sy, cs0, cs1, cs0, sy, cs0, cy], axis=1)
            Ry = tf.reshape(Ry, (-1, 3, 3))
            # z rotation
            thetaz = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cz, sz = tf.cos(thetaz), tf.sin(thetaz)
            Rz = tf.stack([cz, -sz, cs0, sz, cz, cs0, cs0, cs0, cs1], axis=1)
            Rz = tf.reshape(Rz, (-1, 3, 3))
            # whole rotation
            Rxy = tf.matmul(Rx, Ry)
            R = tf.matmul(Rxy, Rz)
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)
            # Apply rotations
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])
        else:
            raise ValueError('Unknown rotation augmentation : ' + self.augment_rotation)

        #######
        # Scale
        #######
        # Choose random scales for each example
        min_s = self.augment_scale_min
        max_s = self.augment_scale_max
        if self.augment_scale_anisotropic:
            s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((num_batches, 1), minval=min_s, maxval=max_s)
        symmetries = []
        for i in range(3):
            if self.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)
        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)
        # Apply scales
        stacked_points = stacked_points * stacked_scales

        #######
        # Noise
        #######
        noise = tf.random_normal(tf.shape(stacked_points), stddev=self.augment_noise)
        stacked_points = stacked_points + noise
        return stacked_points, s, R

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """

        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1),
                                 lambda: tf.zeros((), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before,), dtype=tf.int32)
            inds_in = tf.fill((num_in,), batch_i)
            inds_after = tf.zeros((num_after,), dtype=tf.int32)
            n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

            b_inds += n_inds

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
                                                           tf.TensorShape([None])])

        return batch_inds

    def tf_stack_batch_inds(self, stacks_len):
        # Initiate batch inds tensor
        num_points = tf.reduce_sum(stacks_len)
        max_points = tf.reduce_max(stacks_len)
        batch_inds_0 = tf.zeros((0, max_points), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            # Create this element indices
            element_inds = tf.expand_dims(tf.range(point_i, point_i + stacks_len[batch_i]), axis=0)
            # Pad to right size
            padded_inds = tf.pad(element_inds,
                                 [[0, 0], [0, max_points - stacks_len[batch_i]]],
                                 "CONSTANT",
                                 constant_values=num_points)
            # Concatenate batch indices
            b_inds = tf.concat((b_inds, padded_inds), axis=0)
            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1
            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        fixed_shapes = [tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, None])]
        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=fixed_shapes)

        # Add a last column with shadow neighbor if there is not
        def f1(): return tf.pad(batch_inds, [[0, 0], [0, 1]], "CONSTANT", constant_values=num_points)

        def f2(): return batch_inds

        batch_inds = tf.cond(tf.equal(num_points, max_points * tf.shape(stacks_len)[0]), true_fn=f1, false_fn=f2)
        return batch_inds

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        return neighbors[:, :self.neighborhood_limits[layer]]
