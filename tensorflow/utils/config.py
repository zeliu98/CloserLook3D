import yaml
from easydict import EasyDict as edict

config = edict()
# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #
config.max_epoch = 600
config.batch_size = 16
config.base_learning_rate = 0.002
config.optimizer = 'sgd'
config.momentum = 0.9
config.activation_fn = 'leaky_relu'
config.init = 'xavier'
config.weight_decay = 0.0
config.bn_momentum = 0.99
config.bn_eps = 1e-3
config.grad_norm = 100
config.decay_epoch = 10
config.decay_rate = 0.75
config.epoch_steps = 500
config.validation_size = 50
# ---------------------------------------------------------------------------- #
# Model and Data
# ---------------------------------------------------------------------------- #
config.num_classes = 0
config.num_parts = []
config.in_features_dim = 3
config.bottleneck_ratio = 4
config.depth = 1
config.first_features_dim = 72
config.num_layers = 5
config.global_pooling = 'avg'
config.first_subsampling_dl = 0.02
config.density_parameter = 5.0
config.in_radius = 2.0
# data augmentation
config.augment_scale_anisotropic = True
config.augment_symmetries = [False, False, False]
config.augment_rotation = 'none'
config.augment_scale_min = 0.6
config.augment_scale_max = 1.4
config.augment_noise = 0.002
config.augment_color = 0.8

# ---------------------------------------------------------------------------- #
# io and misc
# ---------------------------------------------------------------------------- #
config.num_gpus = 0
config.num_threads = 0
config.gpus = []
config.load_path = ''
config.print_freq = 10
config.save_freq = 10
config.val_freq = 10
config.log_dir = ''

# ---------------------------------------------------------------------------- #
# Local Aggreagtion Operator
# ---------------------------------------------------------------------------- #
config.local_aggreagtion = 'pospool'
# PosPool
config.pospool = edict()
config.pospool.position_embedding = 'xyz'
config.pospool.reduction = 'mean'
config.pospool.output_conv = False
# Adaptive Weight
config.adaptive_weight = edict()
config.adaptive_weight.local_input_feature = 'dp'
config.adaptive_weight.reduction = 'mean'
config.adaptive_weight.shared_channels = 1
config.adaptive_weight.fc_num = 1
config.adaptive_weight.weight_softmax = False
config.adaptive_weight.output_conv = False
# Point-wise MLP
config.pointwisemlp = edict()
config.pointwisemlp.local_input_feature = 'dp_fj'
config.pointwisemlp.fc_num = 1
config.pointwisemlp.reduction = 'max'
# Pseudo Grid
config.pseudo_grid = edict()
config.pseudo_grid.fixed_kernel_points = 'center'
config.pseudo_grid.KP_influence = 'linear'
config.pseudo_grid.KP_extent = 1.0
config.pseudo_grid.num_kernel_points = 15
config.pseudo_grid.convolution_mode = 'sum'
config.pseudo_grid.output_conv = False


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError(f"key {k} must exist in config.py")
