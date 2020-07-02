"""
Training and evaluating script for scene segmentation with S3DIS dataset
"""
import os
import sys
import time
import pprint
import psutil
import argparse
import subprocess
import numpy as np
import tensorflow as tf

FILE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from datasets import S3DISDataset
from models import SceneSegModel
from utils.config import config, update_config
from utils.average_gradients import average_gradients
from utils.AdamWOptimizer import AdamWeightDecayOptimizer
from utils.logger import setup_logger
from utils.scheduler import StepScheduler
from utils.metrics import AverageMeter, s3dis_metrics, s3dis_subset_metrics, s3dis_voting_metrics


def parse_option():
    parser = argparse.ArgumentParser("Training and evaluating ModelNet40")
    parser.add_argument('--cfg', help='yaml file', type=str)
    parser.add_argument('--gpus', type=int, default=0, nargs='+', help='gpus to use [default: 0]')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate for batch size 8')

    # IO
    parser.add_argument('--log_dir', default='log', help='log dir [default: log]')
    parser.add_argument('--load_path', help='path to a check point file for load')
    parser.add_argument('--print_freq', type=int, help='print frequency')
    parser.add_argument('--save_freq', type=int, help='save frequency')
    parser.add_argument('--val_freq', type=int, help='val frequency')

    # Misc
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')
    parser.add_argument('--save_memory', action='store_true', help='use memory_saving_gradients')

    args, _ = parser.parse_known_args()

    # Update config
    update_config(args.cfg)

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 's3dis', f'{ddir_name}_{int(time.time())}')
    config.load_path = args.load_path
    config.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    config.num_gpus = len(config.gpus)
    if args.num_threads:
        config.num_threads = args.num_threads
    else:
        cpu_count = psutil.cpu_count()
        gpu_count = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        config.num_threads = config.num_gpus * cpu_count // gpu_count
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.print_freq:
        config.print_freq = args.print_freq
    if args.save_freq:
        config.save_freq = args.save_freq
    if args.val_freq:
        config.val_freq = args.val_freq

    # Set manual seed
    tf.set_random_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    # If args.save_memory is True, use gradient-checkpointing to save memory
    if args.save_memory:  # if save memory
        import utils.memory_saving_gradients
        tf.__dict__["gradients"] = utils.memory_saving_gradients.gradients_collection

    return args, config


def training(config):
    with tf.Graph().as_default():
        # Get dataset
        logger.info('==> Preparing datasets...')
        dataset = S3DISDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
        logger.info("==> config.num_classes: {}".format(config.num_classes))
        flat_inputs = dataset.flat_inputs
        train_init_op = dataset.train_init_op
        val_init_op = dataset.val_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Set learning rate and optimizer
        lr_scheduler = StepScheduler('learning_rate',
                                     config.base_learning_rate * config.batch_size * config.num_gpus / 8.0,
                                     config.decay_rate,
                                     config.decay_epoch, config.max_epoch)
        learning_rate = tf.get_variable('learning_rate', [],
                                        initializer=tf.constant_initializer(
                                            config.base_learning_rate * config.batch_size * config.num_gpus / 8.0),
                                        trainable=False)

        if config.optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
        elif config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif config.optimizer == 'adamW':
            optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate,
                                                 weight_decay_rate=config.weight_decay,
                                                 exclude_from_weight_decay=["bias"])

        # -------------------------------------------
        # Get model and loss on multiple GPU devices
        # -------------------------------------------
        # Allocating variables on CPU first will greatly accelerate multi-gpu training.
        # Ref: https://github.com/kuza55/keras-extras/issues/21
        SceneSegModel(flat_inputs[0], is_training_pl, config=config)
        tower_grads = []
        tower_logits = []
        tower_probs = []
        tower_labels = []
        total_loss_gpu = []
        total_segment_loss_gpu = []
        total_weight_loss_gpu = []
        tower_in_batches = []
        tower_point_inds = []
        tower_cloud_inds = []
        for i, igpu in enumerate(config.gpus):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = SceneSegModel(flat_inputs_i, is_training_pl, config=config)
                    logits = model.logits
                    labels = model.labels
                    probs = tf.nn.softmax(model.logits)
                    model.get_loss()
                    losses = tf.get_collection('losses', scope)
                    weight_losses = tf.get_collection('weight_losses', scope)
                    segment_losses = tf.get_collection('segmentation_losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    total_weight_loss = tf.add_n(weight_losses, name='total_weight_loss')
                    total_segment_loss = tf.add_n(segment_losses, name='total_segment_loss')
                    grad_var_list = tf.trainable_variables()
                    if config.optimizer == 'adamW':
                        grads = tf.gradients(total_segment_loss, grad_var_list)
                    else:
                        grads = tf.gradients(total_loss, grad_var_list)
                    grads = list(zip(grads, grad_var_list))
                    tower_grads.append(grads)
                    tower_logits.append(logits)
                    tower_probs.append(probs)
                    tower_labels.append(labels)
                    total_loss_gpu.append(total_loss)
                    total_segment_loss_gpu.append(total_segment_loss)
                    total_weight_loss_gpu.append(total_weight_loss)
                    in_batches = model.inputs['in_batches']
                    point_inds = model.inputs['point_inds']
                    cloud_inds = model.inputs['cloud_inds']
                    tower_in_batches.append(in_batches)
                    tower_point_inds.append(point_inds)
                    tower_cloud_inds.append(cloud_inds)

        # Average losses from multiple GPUs
        total_loss = tf.reduce_mean(total_loss_gpu)
        total_segment_loss = tf.reduce_mean(total_segment_loss_gpu)
        total_weight_loss = tf.reduce_mean(total_weight_loss_gpu)

        # Get training operator
        grads = average_gradients(tower_grads, grad_norm=config.grad_norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads)

        # Add ops to save and restore all the variables.
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SceneSegModel')
        saver = tf.train.Saver(save_vars)

        # Create a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        tfconfig.log_device_placement = False
        sess = tf.Session(config=tfconfig)

        # Initialize variables, resume if needed
        if config.load_path is not None:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, config.load_path)
            logger.info("==> Model loaded in file: %s" % config.load_path)
        else:
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)
            logger.info("==> Init global")

        # Printing model parameters
        all_params = [v for v in tf.trainable_variables() if 'weights' in v.name]
        logger.info("==> All params")
        for param in all_params:
            logger.info(str(param))
        all_params_size = tf.reduce_sum([tf.reduce_prod(v.shape) for v in all_params])
        all_params_size_np = sess.run(all_params_size)
        logger.info("==> Model have {} total Params".format(all_params_size_np))

        ops = {
            'train_init_op': train_init_op,
            'val_init_op': val_init_op,
            'is_training_pl': is_training_pl,
            'tower_logits': tower_logits,
            'tower_probs': tower_probs,
            'tower_labels': tower_labels,
            'tower_in_batches': tower_in_batches,
            'tower_point_inds': tower_point_inds,
            'tower_cloud_inds': tower_cloud_inds,
            'loss': total_loss,
            'segment_loss': total_segment_loss,
            'weight_loss': total_weight_loss,
            'train_op': train_op,
            'learning_rate': learning_rate}

        # For running voting
        validation_probs = [np.zeros((l.shape[0], config.num_classes)) for l in
                            dataset.input_labels['validation']]
        val_proportions = np.zeros(config.num_classes, dtype=np.float32)
        for i, label_value in enumerate(dataset.label_values):
            val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])

        for epoch in range(1, config.max_epoch + 1):
            lr = lr_scheduler.step()
            tic1 = time.time()
            train_one_epoch(sess, ops, epoch, lr)
            tic2 = time.time()
            logger.info("Epoch: {}, total time: {:2f}s, learning rate: {:.5f}, ".format(epoch, tic2 - tic1, lr))
            if epoch % config.val_freq == 0:
                logger.info("==> Validating...")
                val_one_epoch(sess, ops, dataset, validation_probs, val_proportions, epoch)
            if epoch % config.save_freq == 0:
                save_path = saver.save(sess, os.path.join(config.log_dir, "model.ckpt"), global_step=epoch)
                logger.info("==> Model saved in file: {}".format(save_path))
        epoch += 1
        val_one_epoch(sess, ops, dataset, validation_probs, val_proportions, epoch)
        val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=20)
        save_path = saver.save(sess, os.path.join(config.log_dir, "model.ckpt"), global_step=epoch)
        logger.info("==> Model saved in file: {}".format(save_path))
    return save_path


def evaluating(config, save_path, GPUs=0):
    logger.info("==> Start evaluating.........")
    if isinstance(GPUs, list):
        logger.warning("We use the fisrt gpu for evaluating")
        GPUs = [GPUs[0]]
    elif isinstance(GPUs, int):
        GPUs = [GPUs]
    else:
        raise RuntimeError("Check GPUs for evaluate")
    config.num_gpus = 1

    with tf.Graph().as_default():
        logger.info('==> Preparing datasets...')
        dataset = S3DISDataset(config, config.num_threads)
        flat_inputs = dataset.flat_inputs
        val_init_op = dataset.val_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        SceneSegModel(flat_inputs[0], is_training_pl, config=config)
        tower_logits = []
        tower_labels = []
        tower_probs = []
        tower_in_batches = []
        tower_point_inds = []
        tower_cloud_inds = []
        for i, igpu in enumerate(GPUs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = SceneSegModel(flat_inputs_i, is_training_pl, config=config)
                    logits = model.logits
                    labels = model.labels
                    probs = tf.nn.softmax(model.logits)
                    tower_logits.append(logits)
                    tower_probs.append(probs)
                    tower_labels.append(labels)
                    in_batches = model.inputs['in_batches']
                    point_inds = model.inputs['point_inds']
                    cloud_inds = model.inputs['cloud_inds']
                    tower_in_batches.append(in_batches)
                    tower_point_inds.append(point_inds)
                    tower_cloud_inds.append(cloud_inds)

        # Add ops to save and restore all the variables.
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SceneSegModel')
        saver = tf.train.Saver(save_vars)

        # Create a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        tfconfig.log_device_placement = False
        sess = tf.Session(config=tfconfig)

        ops = {'val_init_op': val_init_op,
               'is_training_pl': is_training_pl,
               'tower_logits': tower_logits,
               'tower_probs': tower_probs,
               'tower_labels': tower_labels,
               'tower_in_batches': tower_in_batches,
               'tower_point_inds': tower_point_inds,
               'tower_cloud_inds': tower_cloud_inds,
               }

        # Load the pretrained model
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, save_path)
        logger.info("==> Model loaded in file: %s" % save_path)

        # Evaluating
        logger.info("==> Evaluating Last epoch")
        validation_probs = [np.zeros((l.shape[0], config.num_classes)) for l in
                            dataset.input_labels['validation']]
        val_proportions = np.zeros(config.num_classes, dtype=np.float32)
        for i, label_value in enumerate(dataset.label_values):
            val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])

        val_one_epoch(sess, ops, dataset, validation_probs, val_proportions, 'FINAL')
        val_vote_one_epoch(sess, ops, dataset, 'FINAL', num_votes=20)

    return


def train_one_epoch(sess, ops, epoch, lr):
    """
    One epoch training
    """

    is_training = True

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    weight_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    sess.run(ops['train_init_op'])
    feed_dict = {ops['is_training_pl']: is_training,
                 ops['learning_rate']: lr}

    batch_idx = 0
    end = time.time()
    while True:
        try:
            _, loss, segment_loss, weight_loss = sess.run([ops['train_op'],
                                                           ops['loss'],
                                                           ops['segment_loss'],
                                                           ops['weight_loss']], feed_dict=feed_dict)

            loss_meter.update(loss)
            seg_loss_meter.update(segment_loss)
            weight_loss_meter.update(weight_loss)
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % config.print_freq == 0:
                logger.info(f'Train: [{epoch}][{batch_idx}] '
                            f'T {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                            f'seg loss {seg_loss_meter.val:.3f} ({seg_loss_meter.avg:.3f}) '
                            f'weight loss {weight_loss_meter.val:.3f} ({weight_loss_meter.avg:.3f})')
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break


def val_one_epoch(sess, ops, dataset, validation_probs, val_proportions, epoch):
    """
    One epoch validating
    """

    is_training = False
    feed_dict = {ops['is_training_pl']: is_training}

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95

    loss_meter = AverageMeter()
    weight_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    # Initialise iterator with train data
    sess.run(ops['val_init_op'])

    idx = 0
    predictions = []
    targets = []
    while True:
        try:
            loss, segment_loss, weight_loss, \
            tower_probs, tower_labels, tower_in_batches, tower_point_inds, tower_cloud_inds = sess.run(
                [ops['loss'],
                 ops['segment_loss'],
                 ops['weight_loss'],
                 ops['tower_probs'],
                 ops['tower_labels'],
                 ops['tower_in_batches'],
                 ops['tower_point_inds'],
                 ops['tower_cloud_inds']],
                feed_dict=feed_dict)

            loss_meter.update(loss)
            seg_loss_meter.update(segment_loss)
            weight_loss_meter.update(weight_loss)

            # Stack all validation predictions for each class separately
            for stacked_probs, labels, batches, point_inds, cloud_inds in zip(tower_probs, tower_labels,
                                                                              tower_in_batches, tower_point_inds,
                                                                              tower_cloud_inds):
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]
                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]
                    # Update current probs in whole cloud
                    validation_probs[c_i][inds] = val_smooth * validation_probs[c_i][inds] + (1 - val_smooth) * probs
                    # Stack all prediction for this epoch
                    predictions += [probs]
                    targets += [dataset.input_labels['validation'][c_i][inds]]
            if (idx + 1) % config.print_freq == 0:
                logger.info(f'Val: [{epoch}][{idx}] '
                            f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                            f'seg loss {seg_loss_meter.val:.3f} ({seg_loss_meter.avg:.3f}) '
                            f'weight loss {weight_loss_meter.val:.3f} ({weight_loss_meter.avg:.3f})')
            idx += 1
        except tf.errors.OutOfRangeError:
            break

    IoUs, mIoU = s3dis_subset_metrics(dataset, predictions, targets, val_proportions)
    vote_IoUs, vote_mIoU = s3dis_voting_metrics(dataset, validation_probs, val_proportions)

    logger.info(f'E{epoch} * mIoU {mIoU:.3%} vote_mIoU {vote_mIoU:.3%}')
    logger.info(f'E{epoch} * IoUs {IoUs}')
    logger.info(f'E{epoch} * vote_IoUs {vote_IoUs}')
    return


def val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=20):
    """
    One epoch voting validating
    """

    is_training = False
    feed_dict = {ops['is_training_pl']: is_training}

    # Smoothing parameter for votes
    test_smooth = 0.95

    # Initialise iterator with val data
    sess.run(ops['val_init_op'])

    # Initiate global prediction over test clouds
    nc_model = dataset.num_classes
    val_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32) for l in dataset.input_labels['validation']]
    val_proportions = np.zeros(nc_model, dtype=np.float32)
    for i, label_value in enumerate(dataset.label_values):
        val_proportions[i] = np.sum([np.sum(labels == label_value) for labels in dataset.validation_labels])

    vote_ind = 0
    last_min = -0.5
    while last_min < num_votes:
        try:
            tower_probs, tower_labels, tower_in_batches, tower_point_inds, tower_cloud_inds = sess.run(
                [ops['tower_probs'],
                 ops['tower_labels'],
                 ops['tower_in_batches'],
                 ops['tower_point_inds'],
                 ops['tower_cloud_inds']],
                feed_dict=feed_dict)
            for stacked_probs, labels, batches, point_inds, cloud_inds in zip(tower_probs, tower_labels,
                                                                              tower_in_batches, tower_point_inds,
                                                                              tower_cloud_inds):
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    val_probs[c_i][inds] = test_smooth * val_probs[c_i][inds] + (1 - test_smooth) * probs
        except:
            new_min = np.min(dataset.min_potentials['validation'])
            logger.info('Step {:3d}, end. Min potential = {:.1f}'.format(vote_ind, new_min))
            if last_min + 1 < new_min:
                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                logger.info('==> Confusion on sub clouds')
                IoUs, mIoU = s3dis_voting_metrics(dataset, val_probs, val_proportions)
                logger.info(f'E{epoch} S{vote_ind} * mIoU {mIoU:.3%}')

                if int(np.ceil(new_min)) % 2 == 0:
                    # Project predictions
                    v = int(np.floor(new_min))
                    logger.info('Reproject True Vote #{:d}'.format(v))
                    files = dataset.train_files
                    i_val = 0
                    proj_probs = []
                    for i, file_path in enumerate(files):
                        if dataset.all_splits[i] == dataset.validation_split:
                            # Reproject probs on the evaluations points
                            probs = val_probs[i_val][dataset.validation_proj[i_val], :]
                            proj_probs += [probs]
                            i_val += 1
                    # Show vote results
                    logger.info('==> Confusion on full clouds')
                    IoUs, mIoU = s3dis_metrics(dataset, proj_probs)
                    logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
                    logger.info(f'E{epoch} V{v} * IoUs {IoUs}')

            sess.run(ops['val_init_op'])
            vote_ind += 1

    # Project predictions
    logger.info('Reproject True Vote Last')
    files = dataset.train_files
    i_val = 0
    proj_probs = []
    for i, file_path in enumerate(files):
        if dataset.all_splits[i] == dataset.validation_split:
            # Reproject probs on the evaluations points
            probs = val_probs[i_val][dataset.validation_proj[i_val], :]
            proj_probs += [probs]
            i_val += 1
    # Show vote results
    logger.info('==> Confusion on full clouds')
    IoUs, mIoU = s3dis_metrics(dataset, proj_probs)
    logger.info(f'E{epoch} VLast * mIoU {mIoU:.3%}')
    logger.info(f'E{epoch} VLast * IoUs {IoUs}')

    return


if __name__ == "__main__":
    _, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="s3dis")
    logger.info(pprint.pformat(config))
    save_path = training(config)
    evaluating(config, save_path, config.gpus[0])
