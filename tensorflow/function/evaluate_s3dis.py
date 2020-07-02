"""
Evaluating script for scene segmentation with S3DIS dataset
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

from datasets import S3DISDataset
from models import SceneSegModel
from utils.config import config, update_config
from utils.logger import setup_logger
from utils.metrics import s3dis_metrics, s3dis_subset_metrics, s3dis_voting_metrics


def parse_option():
    parser = argparse.ArgumentParser("Evaluating ModelNet40")
    parser.add_argument('--cfg', help='yaml file', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use [default: 0]')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate for batch size 8')

    # IO
    parser.add_argument('--log_dir', default='log_eval', help='log dir [default: log]')
    parser.add_argument('--load_path', help='path to a check point file for load')

    # Misc
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args, _ = parser.parse_known_args()

    # Update config
    update_config(args.cfg)

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 's3dis', f'{ddir_name}_{int(time.time())}')
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
        dataset = S3DISDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
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


def val_one_epoch(sess, ops, dataset, validation_probs, val_proportions, epoch):
    """
    One epoch validating
    """

    is_training = False
    feed_dict = {ops['is_training_pl']: is_training}

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95

    # Initialise iterator with train data
    sess.run(ops['val_init_op'])

    idx = 0
    predictions = []
    targets = []
    while True:
        try:
            tower_probs, tower_labels, tower_in_batches, tower_point_inds, tower_cloud_inds = sess.run(
                [ops['tower_probs'],
                 ops['tower_labels'],
                 ops['tower_in_batches'],
                 ops['tower_point_inds'],
                 ops['tower_cloud_inds']],
                feed_dict=feed_dict)

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
    args, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="s3dis_eval")
    logger.info(pprint.pformat(config))
    evaluating(config, config.load_path, args.gpu)
