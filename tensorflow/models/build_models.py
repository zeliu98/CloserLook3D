import tensorflow as tf

from .heads import resnet_classification_head, resnet_scene_segmentation_head, resnet_multi_part_segmentation_head
from .backbone import resnet_backbone


class PartSegModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['super_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['object_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']
            self.shape_label = self.inputs['super_labels']

        with tf.variable_scope('PartSegModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']
            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, activation_fn=config.activation_fn,
                                weight_decay=config.weight_decay, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits_with_point_label, self.logits_all_shapes = \
                resnet_multi_part_segmentation_head(config,
                                                    self.inputs, F,
                                                    base_fdim=fdim,
                                                    is_training=is_training,
                                                    init=config.init,
                                                    weight_decay=config.weight_decay,
                                                    activation_fn=config.activation_fn,
                                                    bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_loss(self):
        cross_entropy = 0.0
        for i_shape in range(self.config.num_classes):
            logits_i, point_labels_i = self.logits_with_point_label[i_shape]
            cross_entropy_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=point_labels_i,
                                                                             logits=logits_i,
                                                                             name=f'cross_entropy_shape{i_shape}')
            cross_entropy += tf.reduce_sum(cross_entropy_i)

        num_inst = tf.shape(self.inputs['point_labels'])[0]
        self.loss = cross_entropy / tf.cast(num_inst, dtype=tf.float32)
        tf.add_to_collection('losses', self.loss)
        tf.add_to_collection('segmentation_losses', self.loss)


class ClassificationModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            ind = 3 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['object_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['labels']

        with tf.variable_scope('ClassificationModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits = resnet_classification_head(config, self.inputs, F[-1], base_fdim=fdim,
                                                     is_training=is_training, pooling=config.global_pooling,
                                                     init=config.init, weight_decay=config.weight_decay,
                                                     activation_fn=config.activation_fn, bn=True,
                                                     bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_loss(self):
        labels = tf.one_hot(indices=self.labels, depth=self.config.num_classes)
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits,
                                                        label_smoothing=0.2,
                                                        scope='cross_entropy')  # be care of label smoothing

        self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        tf.add_to_collection('losses', self.loss)
        tf.add_to_collection('classification_losses', self.loss)


class SceneSegModel(object):
    def __init__(self, flat_inputs, is_training, config):
        self.config = config
        self.is_training = is_training

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']

        with tf.variable_scope('SceneSegModel'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)
            self.logits = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
                                                         is_training=is_training, init=config.init,
                                                         weight_decay=config.weight_decay,
                                                         activation_fn=config.activation_fn,
                                                         bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

    def get_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputs['point_labels'],
                                                                       logits=self.logits,
                                                                       name='cross_entropy')
        cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        self.loss = cross_entropy
        tf.add_to_collection('losses', self.loss)
        tf.add_to_collection('segmentation_losses', self.loss)
