"""
Training and evaluating script for part segmentation with PartNet dataset
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
from sklearn.neighbors import KDTree

FILE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from datasets import PartNetDataset
from models import PartSegModel
from utils.config import config, update_config
from utils.average_gradients import average_gradients
from utils.AdamWOptimizer import AdamWeightDecayOptimizer
from utils.logger import setup_logger
from utils.scheduler import StepScheduler
from utils.metrics import AverageMeter, partnet_metrics
from utils.ply import read_ply


def parse_option():
    parser = argparse.ArgumentParser("Training and evaluating PartNet")
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
    parser.add_argument('--save_memory', action='store_true', help='use memory_saving_gradients')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args, _ = parser.parse_known_args()

    # Update config
    update_config(args.cfg)

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'partnet', f'{ddir_name}_{int(time.time())}')
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
        config.save_freq = args.val_freq

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
        dataset = PartNetDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
        config.num_parts = dataset.num_parts
        print("config.num_classes: {}".format(config.num_classes))
        print("config.num_parts: {}".format(config.num_parts))

        flat_inputs = dataset.flat_inputs
        train_init_op = dataset.train_init_op
        val_init_op = dataset.val_init_op
        test_init_op = dataset.test_init_op
        val_vote_init_op = dataset.val_vote_init_op
        test_vote_init_op = dataset.test_vote_init_op

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
            optimizer = AdamWeightDecayOptimizer(learning_rate=config.base_learning_rate / 8.0,
                                                 weight_decay_rate=config.weight_decay,
                                                 exclude_from_weight_decay=["bias"])
        else:
            raise NotImplementedError

        # -------------------------------------------
        # Get model and loss on multiple GPU devices
        # -------------------------------------------
        # Allocating variables on CPU first will greatly accelerate multi-gpu training.
        # Ref: https://github.com/kuza55/keras-extras/issues/21
        PartSegModel(flat_inputs[0], is_training_pl, config=config)
        tower_grads = []
        tower_logits_with_point_label = []
        tower_logits_all_shapes = []
        tower_labels = []
        total_loss_gpu = []
        total_segment_loss_gpu = []
        total_weight_loss_gpu = []
        tower_super_labels = []
        tower_object_inds = []
        tower_in_batches = []
        for i, igpu in enumerate(config.gpus):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = PartSegModel(flat_inputs_i, is_training_pl, config=config)
                    logits_with_point_label = model.logits_with_point_label
                    logits_all_shapes = model.logits_all_shapes
                    labels = model.labels
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
                    tower_logits_with_point_label.append(logits_with_point_label)
                    tower_logits_all_shapes.append(logits_all_shapes)
                    tower_labels.append(labels)
                    total_loss_gpu.append(total_loss)
                    total_segment_loss_gpu.append(total_segment_loss)
                    total_weight_loss_gpu.append(total_weight_loss)
                    super_labels = model.inputs['super_labels']
                    object_inds = model.inputs['object_inds']
                    in_batches = model.inputs['in_batches']
                    tower_super_labels.append(super_labels)
                    tower_object_inds.append(object_inds)
                    tower_in_batches.append(in_batches)

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
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PartSegModel')
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
            logger.info("Model loaded in file: %s" % config.load_path)
        else:
            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)
            logger.info("init global")

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
            'val_vote_init_op': val_vote_init_op,
            'test_init_op': test_init_op,
            'test_vote_init_op': test_vote_init_op,
            'is_training_pl': is_training_pl,
            'tower_logits_with_point_label': tower_logits_with_point_label,
            'tower_logits_all_shapes': tower_logits_all_shapes,
            'tower_labels': tower_labels,
            'tower_super_labels': tower_super_labels,
            'tower_object_inds': tower_object_inds,
            'tower_in_batches': tower_in_batches,
            'loss': total_loss,
            'segment_loss': total_segment_loss,
            'weight_loss': total_weight_loss,
            'train_op': train_op,
            'learning_rate': learning_rate}

        for epoch in range(1, config.max_epoch + 1):
            lr = lr_scheduler.step()
            tic1 = time.time()

            train_one_epoch(sess, ops, epoch, lr)
            tic2 = time.time()
            logger.info("Epoch: {} total time: {:2f}s, learning rate: {:.5f}".format(epoch, tic2 - tic1, lr))
            if epoch % config.val_freq == 0:
                logger.info("==> Validating...")
                val_one_epoch(sess, ops, dataset, epoch, 'val')
                val_one_epoch(sess, ops, dataset, epoch, 'test')
            if epoch % config.save_freq == 0:
                save_path = saver.save(sess, os.path.join(config.log_dir, "model.ckpt"), global_step=epoch)
                logger.info("==> Model saved in file: {}".format(save_path))
        epoch += 1
        val_one_epoch(sess, ops, dataset, epoch, 'val')
        val_one_epoch(sess, ops, dataset, epoch, 'test')
        val_vote_one_epoch(sess, ops, dataset, epoch, 'val', num_votes=10)
        val_vote_one_epoch(sess, ops, dataset, epoch, 'test', num_votes=10)
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
        dataset = PartNetDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
        config.num_parts = dataset.num_parts
        print("config.num_classes: {}".format(config.num_classes))
        print("config.num_parts: {}".format(config.num_parts))
        flat_inputs = dataset.flat_inputs
        val_init_op = dataset.val_init_op
        test_init_op = dataset.test_init_op
        val_vote_init_op = dataset.val_vote_init_op
        test_vote_init_op = dataset.test_vote_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        PartSegModel(flat_inputs[0], is_training_pl, config=config)
        tower_logits_with_point_label = []
        tower_logits_all_shapes = []
        tower_labels = []
        tower_super_labels = []
        tower_object_inds = []
        tower_in_batches = []
        for i, igpu in enumerate(GPUs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = PartSegModel(flat_inputs_i, is_training_pl, config=config)
                    logits_with_point_label = model.logits_with_point_label
                    logits_all_shapes = model.logits_all_shapes
                    labels = model.labels
                    tower_logits_with_point_label.append(logits_with_point_label)
                    tower_logits_all_shapes.append(logits_all_shapes)
                    tower_labels.append(labels)
                    super_labels = model.inputs['super_labels']
                    object_inds = model.inputs['object_inds']
                    in_batches = model.inputs['in_batches']
                    tower_super_labels.append(super_labels)
                    tower_object_inds.append(object_inds)
                    tower_in_batches.append(in_batches)

        # Add ops to save and restore all the variables.
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PartSegModel')
        saver = tf.train.Saver(save_vars)

        # Create a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        tfconfig.log_device_placement = False
        sess = tf.Session(config=tfconfig)

        ops = {'val_init_op': val_init_op,
               'test_init_op': test_init_op,
               'val_vote_init_op': val_vote_init_op,
               'test_vote_init_op': test_vote_init_op,
               'is_training_pl': is_training_pl,
               'tower_logits_with_point_label': tower_logits_with_point_label,
               'tower_logits_all_shapes': tower_logits_all_shapes,
               'tower_labels': tower_labels,
               'tower_super_labels': tower_super_labels,
               'tower_object_inds': tower_object_inds,
               'tower_in_batches': tower_in_batches}

        # Load the pretrained model
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, save_path)
        logger.info("Model loaded in file: %s" % save_path)

        # Evaluating
        logger.info("==> Evaluating Last epoch")
        val_one_epoch(sess, ops, dataset, 'FINAL', 'val')
        val_one_epoch(sess, ops, dataset, 'FINAL', 'test')
        val_vote_one_epoch(sess, ops, dataset, 'FINAL', 'val', num_votes=10)
        val_vote_one_epoch(sess, ops, dataset, 'FINAL', 'test', num_votes=10)

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
                                                           ops['weight_loss']],
                                                          feed_dict=feed_dict)
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


def val_one_epoch(sess, ops, dataset, epoch, split):
    """
    One epoch validating
    """

    is_training = False

    sess.run(ops[f'{split}_init_op'])
    feed_dict = {ops['is_training_pl']: is_training}

    preds = []
    targets = []
    objects = []
    obj_inds = []
    idx = 0
    while True:
        try:
            tower_logits_all_shapes, tower_labels, \
            tower_object_labels, tower_o_inds, tower_batches = sess.run([ops['tower_logits_all_shapes'],
                                                                         ops['tower_labels'],
                                                                         ops['tower_super_labels'],
                                                                         ops['tower_object_inds'],
                                                                         ops['tower_in_batches']],
                                                                        feed_dict=feed_dict)
            # Get predictions and labels per instance
            for logits_all_shapes, labels, object_labels, o_inds, batches in zip(tower_logits_all_shapes, tower_labels,
                                                                                 tower_object_labels, tower_o_inds,
                                                                                 tower_batches):
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]
                    # Get prediction (only for the concerned parts)
                    obj = object_labels[b[0]]
                    pred = logits_all_shapes[obj][b]
                    # Stack all results
                    objects += [obj]
                    obj_inds += [o_inds[b_i]]
                    preds += [pred]
                    targets += [labels[b]]
            idx += 1
        except tf.errors.OutOfRangeError:
            break

    msIoU, mpIoU, mmsIoU, mmpIoU = partnet_metrics(dataset.num_classes, dataset.num_parts,
                                                   objects, preds, targets)

    logger.info(f'E{epoch} {split} * mmsIoU {mmsIoU:.3%} mmpIoU {mmpIoU:.3%}')
    logger.info(f'E{epoch} {split} * msIoU {msIoU}')
    logger.info(f'E{epoch} {split} * mpIoU {mpIoU}')

    return


def val_vote_one_epoch(sess, ops, dataset, epoch, split, num_votes=10):
    """
    One epoch voting validating
    """

    is_training = False

    original_labels, original_points, \
    projection_inds, average_predictions = prepare_testing_structure(dataset, split)
    feed_dict = {ops['is_training_pl']: is_training}

    for v in range(num_votes):
        if v == 0:
            sess.run(ops[f'{split}_init_op'])
        else:
            sess.run(ops[f'{split}_vote_init_op'])
        all_predictions = []
        all_obj_inds = []
        all_objects = []
        while True:
            try:
                tower_logits_all_shapes, tower_labels, \
                tower_object_labels, tower_o_inds, tower_batches = sess.run([ops['tower_logits_all_shapes'],
                                                                             ops['tower_labels'],
                                                                             ops['tower_super_labels'],
                                                                             ops['tower_object_inds'],
                                                                             ops['tower_in_batches']],
                                                                            feed_dict=feed_dict)

                for logits_all_shapes, labels, object_labels, o_inds, batches in zip(tower_logits_all_shapes,
                                                                                     tower_labels,
                                                                                     tower_object_labels,
                                                                                     tower_o_inds,
                                                                                     tower_batches):
                    max_ind = np.max(batches)
                    for b_i, b in enumerate(batches):
                        # Eliminate shadow indices
                        b = b[b < max_ind - 0.5]
                        # Get prediction (only for the concerned parts)
                        obj = object_labels[b[0]]
                        pred = logits_all_shapes[obj][b]
                        # Stack all results
                        all_objects += [obj]
                        all_obj_inds += [o_inds[b_i]]
                        all_predictions += [pred]
            except tf.errors.OutOfRangeError:
                break

        true_num_test = len(all_predictions)
        num_test = dataset.num_test if split == 'test' else dataset.num_val
        vote_objects = [-1] * num_test
        # Project predictions on original point clouds
        for i, probs in enumerate(all_predictions):
            # Interpolate prediction from current positions to original points
            obj_i = all_obj_inds[i]
            proj_predictions = probs[projection_inds[obj_i]]
            vote_objects[obj_i] = all_objects[i]
            # Average prediction across votes
            average_predictions[obj_i] = average_predictions[obj_i] + \
                                         (proj_predictions - average_predictions[obj_i]) / (v + 1)

        if true_num_test != num_test:
            logger.warning("{} using {}/{} data, "
                           "this may be caused by multi-gpu testing".format(split, true_num_test, num_test))
            vote_preds = average_predictions[:true_num_test]
            vote_targets = original_labels[:true_num_test]
        else:
            vote_preds = average_predictions
            vote_targets = original_labels

        msIoU, mpIoU, mmsIoU, mmpIoU = partnet_metrics(dataset.num_classes, dataset.num_parts,
                                                       vote_objects, vote_preds, vote_targets)
        logger.info(f'E{epoch} V{v} {split} * mmsIoU {mmsIoU:.3%} mmpIoU {mmpIoU:.3%}')
        logger.info(f'E{epoch} V{v} {split} * msIoU {msIoU}')
        logger.info(f'E{epoch} V{v} {split} * mpIoU {mpIoU}')
    return


def prepare_testing_structure(dataset, split):
    logger.info('==> Preparing test structures')
    t1 = time.time()

    # Collect original test file names
    original_path = os.path.join(dataset.path, f'{split}_ply')
    test_names = [f[:-4] for f in os.listdir(original_path) if f[-4:] == '.ply']
    test_names = np.sort(test_names)

    original_labels = []
    original_points = []
    projection_inds = []
    for i, cloud_name in enumerate(test_names):
        # Read data in ply file
        data = read_ply(os.path.join(original_path, cloud_name + '.ply'))
        points = np.vstack((data['x'], data['y'], data['z'])).T
        original_labels += [data['label']]
        original_points += [points]

        # Create tree structure to compute neighbors
        if split == 'val':
            tree = KDTree(dataset.input_points['validation'][i])
        else:
            tree = KDTree(dataset.input_points['test'][i])
        projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

    # Initiate result containers
    average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]
    t2 = time.time()
    print('Done in {:.1f} s\n'.format(t2 - t1))
    return original_labels, original_points, projection_inds, average_predictions


if __name__ == "__main__":
    _, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="partnet")
    logger.info(pprint.pformat(config))
    save_path = training(config)
    evaluating(config, save_path, config.gpus[0])
