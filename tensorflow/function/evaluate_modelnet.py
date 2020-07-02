"""
Evaluating script for 3D shape classification with ModelNet40 dataset
"""
import os
import sys
import time
import pprint
import psutil
import argparse
import numpy as np
import tensorflow as tf

FILE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from datasets import ModelNetDataset
from models import ClassificationModel
from utils.config import config, update_config
from utils.logger import setup_logger
from utils.metrics import classification_metrics


def parse_option():
    parser = argparse.ArgumentParser("Evaluating ModelNet40")
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
    config.log_dir = os.path.join(args.log_dir, 'modelnet', f'{ddir_name}_{int(time.time())}')
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
        dataset = ModelNetDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
        flat_inputs = dataset.flat_inputs
        val_init_op = dataset.val_init_op
        val_vote_init_op = dataset.val_vote_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        ClassificationModel(flat_inputs[0], is_training_pl, config=config)
        tower_logits = []
        tower_labels = []
        tower_object_inds = []
        for i, igpu in enumerate(GPUs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = ClassificationModel(flat_inputs_i, is_training_pl, config=config)
                    logits = model.logits
                    labels = model.labels
                    tower_logits.append(logits)
                    tower_labels.append(labels)
                    object_inds = model.inputs['object_inds']
                    tower_object_inds.append(object_inds)

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ClassificationModel')
        saver = tf.train.Saver(save_vars)

        # Create a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        tfconfig.log_device_placement = False
        sess = tf.Session(config=tfconfig)

        ops = {'val_init_op': val_init_op,
               'val_vote_init_op': val_vote_init_op,
               'is_training_pl': is_training_pl,
               'tower_logits': tower_logits,
               'tower_labels': tower_labels,
               'tower_object_inds': tower_object_inds}

        # Load the pretrained model
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, save_path)
        logger.info("==> Model loaded in file: %s" % save_path)

        # Evaluating
        logger.info("==> Evaluating Last epoch")
        val_one_epoch(sess, ops, dataset, 'LastEpoch')
        val_vote_one_epoch(sess, ops, dataset, 'LastEpoch', num_votes=100)

    return


def val_one_epoch(sess, ops, dataset, epoch):
    """
    One epoch validating
    """

    is_training = False

    sess.run(ops['val_init_op'])
    feed_dict = {ops['is_training_pl']: is_training}

    preds = []
    targets = []
    obj_inds = []
    idx = 0
    while True:
        try:
            tower_logits, tower_labels, tower_o_inds = sess.run(
                [ops['tower_logits'],
                 ops['tower_labels'],
                 ops['tower_object_inds'], ],
                feed_dict=feed_dict)

            for logits, labels, inds in zip(tower_logits, tower_labels, tower_o_inds):
                preds += [logits]
                targets += [labels]
                obj_inds += [inds]

            idx += 1
        except tf.errors.OutOfRangeError:
            break

    # Stack all validation predictions
    preds = np.vstack(preds)
    targets = np.hstack(targets)
    obj_inds = np.hstack(obj_inds)
    true_num_test = preds.shape[0]

    val_preds = np.zeros((len(dataset.input_labels['validation']), dataset.num_classes))
    val_targets = np.zeros((len(dataset.input_labels['validation'])))
    val_preds[obj_inds] = preds
    val_targets[obj_inds] = targets
    if true_num_test != dataset.num_test:
        logger.warning("Validating using {}/{} data, "
                       "this may be caused by multi-gpu testing".format(true_num_test, dataset.num_test))
        val_preds = val_preds[:true_num_test]
        val_targets = val_targets[:true_num_test]

    acc, avg_class_acc = classification_metrics(val_preds, val_targets, dataset.num_classes)

    logger.info(f'E{epoch} * Acc {acc:.3%} CAcc {avg_class_acc:.3%}')

    return acc, avg_class_acc


def val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=10):
    """
    One epoch voting validating
    """

    is_training = False
    average_preds = np.zeros((dataset.num_test, dataset.num_classes))
    original_labels = dataset.input_labels['validation']
    feed_dict = {ops['is_training_pl']: is_training}
    for v in range(num_votes):
        if v == 0:
            sess.run(ops['val_init_op'])
        else:
            sess.run(ops['val_vote_init_op'])
        preds = []
        targets = []
        obj_inds = []
        while True:
            try:
                tower_logits, tower_labels, tower_o_inds = sess.run([ops['tower_logits'],
                                                                     ops['tower_labels'],
                                                                     ops['tower_object_inds']],
                                                                    feed_dict=feed_dict)

                for logits, labels, inds in zip(tower_logits, tower_labels, tower_o_inds):
                    preds += [logits]
                    targets += [labels]
                    obj_inds += [inds]

            except tf.errors.OutOfRangeError:
                break

        # Stack all validation predictions
        preds = np.vstack(preds)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)
        true_num_test = preds.shape[0]
        average_preds[obj_inds] = average_preds[obj_inds] + (preds - average_preds[obj_inds]) / (v + 1)

        if true_num_test != dataset.num_test:
            logger.warning("Validating using {}/{} data, "
                           "this may be caused by multi-gpu testing".format(true_num_test, dataset.num_test))
            vote_preds = average_preds[:true_num_test]
            vote_targets = original_labels[:true_num_test]
        else:
            vote_preds = average_preds
            vote_targets = original_labels

        if np.any(vote_targets[obj_inds] != targets):
            raise ValueError('wrong object indices')

        acc, avg_class_acc = classification_metrics(vote_preds, vote_targets, dataset.num_classes)

        logger.info(f'E{epoch} V{v} * Acc {acc:.3%} CAcc {avg_class_acc:.3%}')

    return acc, avg_class_acc


if __name__ == "__main__":
    args, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="modelnet40_eval")
    logger.info(pprint.pformat(config))
    evaluating(config, config.load_path, args.gpu)
