import yaml
from easydict import EasyDict as edict

config = edict()
# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
config.epochs = 600
config.start_epoch = 1
config.base_learning_rate = 0.01
config.lr_scheduler = 'step'  # step,cosine
config.optimizer = 'sgd'
config.warmup_epoch = 5
config.warmup_multiplier = 100
config.lr_decay_steps = 20
config.lr_decay_rate = 0.7
config.weight_decay = 0
config.momentum = 0.9
config.grid_clip_norm = -1
# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
config.backbone = 'resnet'
config.head = 'resnet_cls'
config.radius = 0.05
config.sampleDl = 0.02
config.density_parameter = 5.0
config.nsamples = []
config.npoints = []
config.width = 144
config.depth = 2
config.bottleneck_ratio = 2
config.bn_momentum = 0.1

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
config.datasets = 'modelnet40'
config.data_root = ''
config.num_classes = 0
config.num_parts = 0
config.input_features_dim = 3
config.batch_size = 32
config.num_points = 5000
config.num_classes = 40
config.num_workers = 4
# data augmentation
config.x_angle_range = 0.0
config.y_angle_range = 0.0
config.z_angle_range = 0.0
config.scale_low = 2. / 3.
config.scale_high = 3. / 2.
config.noise_std = 0.01
config.noise_clip = 0.05
config.translate_range = 0.2
config.color_drop = 0.2
config.augment_symmetries = [0, 0, 0]
# scene segmentation related
config.in_radius = 2.0
config.num_steps = 500

# ---------------------------------------------------------------------------- #
# io and misc
# ---------------------------------------------------------------------------- #
config.load_path = ''
config.print_freq = 10
config.save_freq = 10
config.val_freq = 10
config.log_dir = 'log'
config.local_rank = 0
config.amp_opt_level = ''
config.rng_seed = 0

# ---------------------------------------------------------------------------- #
# Local Aggregation options
# ---------------------------------------------------------------------------- #
config.local_aggregation_type = 'pospool'  # pospool, continuous_conv
# PosPool
config.pospool = edict()
config.pospool.position_embedding = 'xyz'
config.pospool.reduction = 'sum'
config.pospool.output_conv = False
# adaptive_weight
config.adaptive_weight = edict()
config.adaptive_weight.weight_type = 'dp'  # dp, df, dp_df, fj, dp_fj, fi_df, dp_fi_df, rscnn
config.adaptive_weight.num_mlps = 1
config.adaptive_weight.shared_channels = 1
config.adaptive_weight.weight_softmax = False
config.adaptive_weight.reduction = 'avg'  # sum_conv, max_conv, mean_conv
config.adaptive_weight.output_conv = False
# pointwisemlp
config.pointwisemlp = edict()
config.pointwisemlp.feature_type = 'dp_fj'  # dp_fj, fi_df, dp_fi_df
config.pointwisemlp.num_mlps = 1
config.pointwisemlp.reduction = 'max'
# pseudo_grid
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
                raise ValueError(f"{k} key must exist in config.py")
