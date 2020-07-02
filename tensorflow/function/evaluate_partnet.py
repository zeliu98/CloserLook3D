"""
Evaluating script for part segmentation with PartNet dataset
"""
import os
import sys
import time
import pprint
import psutil
import argparse
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
from utils.logger import setup_logger
from utils.metrics import partnet_metrics
from utils.ply import read_ply


def parse_option():
    parser = argparse.ArgumentParser("Evaluating PartNet")
    parser.add_argument('--cfg', help='yaml file', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use [default: 0]')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')

    # IO
    parser.add_argument('--log_dir', default='log_eval', help='log dir [default: log]')
    parser.add_argument('--load_path', help='path to a check point file for load')

    # Misc
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args, _ = parser.parse_known_args()

    # Update config
    update_config(args.cfg)

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'partnet', f'{ddir_name}_{int(time.time())}')
    config.load_path = args.load_path
    if args.num_threads:
        config.num_threads = args.num_threads
    else:
        cpu_count = psutil.cpu_count()
        config.num_threads = cpu_count
    if args.batch_size:
        config.batch_size = args.batch_size

    # Set manual seed
    tf.set_random_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


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
    args, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="partnet_eval")
    logger.info(pprint.pformat(config))
    evaluating(config, config.load_path, args.gpu)
