"""
Training and evaluating script for 3D shape classification with ModelNet40 dataset
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

from datasets import ModelNetDataset
from models import ClassificationModel
from utils.config import config, update_config
from utils.average_gradients import average_gradients
from utils.AdamWOptimizer import AdamWeightDecayOptimizer
from utils.logger import setup_logger
from utils.scheduler import StepScheduler
from utils.metrics import AverageMeter, classification_metrics


def parse_option():
    parser = argparse.ArgumentParser("Training and evaluating ModelNet40")
    parser.add_argument('--cfg', help='yaml file', type=str)
    parser.add_argument('--gpus', type=int, default=0, nargs='+', help='gpus to use [default: 0]')
    parser.add_argument('--num_threads', type=int, default=4, help='num of threads to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate for batch size 16')

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
    config.log_dir = os.path.join(args.log_dir, 'modelnet', f'{ddir_name}_{int(time.time())}')
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
        dataset = ModelNetDataset(config, config.num_threads)
        config.num_classes = dataset.num_classes
        print("==> config.num_classes: {}".format(config.num_classes))
        flat_inputs = dataset.flat_inputs
        train_init_op = dataset.train_init_op
        val_init_op = dataset.val_init_op
        val_vote_init_op = dataset.val_vote_init_op

        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Set learning rate and optimizer
        lr_scheduler = StepScheduler('learning_rate',
                                     config.base_learning_rate * config.batch_size * config.num_gpus / 16.0,
                                     config.decay_rate,
                                     config.decay_epoch, config.max_epoch)
        learning_rate = tf.get_variable('learning_rate', [],
                                        initializer=tf.constant_initializer(
                                            config.base_learning_rate * config.batch_size * config.num_gpus / 16.0),
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
        ClassificationModel(flat_inputs[0], is_training_pl, config=config)
        tower_grads = []
        tower_logits = []
        tower_labels = []
        tower_object_inds = []
        total_loss_gpu = []
        total_classification_loss_gpu = []
        total_weight_loss_gpu = []
        for i, igpu in enumerate(config.gpus):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                with tf.device('/gpu:%d' % (igpu)), tf.name_scope('gpu_%d' % (igpu)) as scope:
                    flat_inputs_i = flat_inputs[i]
                    model = ClassificationModel(flat_inputs_i, is_training_pl, config=config)
                    logits = model.logits
                    labels = model.labels
                    model.get_loss()
                    losses = tf.get_collection('losses', scope)
                    weight_losses = tf.get_collection('weight_losses', scope)
                    classification_losses = tf.get_collection('classification_losses', scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    total_weight_loss = tf.add_n(weight_losses, name='total_weight_loss')
                    total_classification_loss = tf.add_n(classification_losses, name='total_classification_loss')
                    grad_var_list = tf.trainable_variables()
                    if config.optimizer == 'adamW':
                        grads = tf.gradients(total_classification_loss, grad_var_list)
                    else:
                        grads = tf.gradients(total_loss, grad_var_list)
                    grads = list(zip(grads, grad_var_list))
                    tower_grads.append(grads)
                    tower_logits.append(logits)
                    tower_labels.append(labels)
                    object_inds = model.inputs['object_inds']
                    tower_object_inds.append(object_inds)
                    total_loss_gpu.append(total_loss)
                    total_classification_loss_gpu.append(total_classification_loss)
                    total_weight_loss_gpu.append(total_weight_loss)

        # Average losses from multiple GPUs
        total_loss = tf.reduce_mean(total_loss_gpu)
        total_classification_loss = tf.reduce_mean(total_classification_loss_gpu)
        total_weight_loss = tf.reduce_mean(total_weight_loss_gpu)

        # Get training operator
        grads = average_gradients(tower_grads, grad_norm=config.grad_norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads)

        # Add ops to save and restore all the variables.
        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ClassificationModel')
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

        ops = {'train_init_op': train_init_op,
               'val_init_op': val_init_op,
               'val_vote_init_op': val_vote_init_op,
               'is_training_pl': is_training_pl,
               'tower_logits': tower_logits,
               'tower_labels': tower_labels,
               'tower_object_inds': tower_object_inds,
               'loss': total_loss,
               'classification_loss': total_classification_loss,
               'weight_loss': total_weight_loss,
               'train_op': train_op,
               'learning_rate': learning_rate}

        best_acc = 0
        best_epoch = 0
        best_vote_acc = 0
        best_vote_epoch = 0
        for epoch in range(1, config.max_epoch + 1):
            lr = lr_scheduler.step()
            tic1 = time.time()
            train_one_epoch(sess, ops, epoch, lr)
            tic2 = time.time()
            logger.info("Epoch: {}, total time: {:2f}s, learning rate: {:.5f}, "
                        "best acc: {:3%}/{}, best vote acc: {:3%}/{}".format(epoch, tic2 - tic1, lr,
                                                                             best_acc, best_epoch,
                                                                             best_vote_acc, best_vote_epoch))
            logger.info("==> Validating...")
            acc, cls_acc = val_one_epoch(sess, ops, dataset, epoch)
            best_acc = max(best_acc, acc)
            best_epoch = epoch if (best_acc == acc) else best_epoch
            if epoch % config.val_freq == 0:
                logger.info("==> Voting Validating...")
                vote_acc, vote_cls_acc = val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=10)
                if vote_acc > best_vote_acc:
                    best_vote_acc = vote_acc
                    best_vote_epoch = epoch
                    save_path = saver.save(sess, os.path.join(config.log_dir, "best.ckpt"), global_step=epoch)
                    logger.info("==> Model saved in file: {}".format(save_path))

            if epoch % config.save_freq == 0:
                save_path = saver.save(sess, os.path.join(config.log_dir, "model.ckpt"), global_step=epoch)
                logger.info("==> Model saved in file: {}".format(save_path))
        epoch += 1
        val_vote_one_epoch(sess, ops, dataset, epoch, num_votes=10)
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
        dataset = ModelNetDataset(config, config.num_threads)
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


def train_one_epoch(sess, ops, epoch, lr):
    """
    One epoch training
    """

    is_training = True

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    weight_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    sess.run(ops['train_init_op'])
    feed_dict = {ops['is_training_pl']: is_training,
                 ops['learning_rate']: lr}

    batch_idx = 0
    end = time.time()
    while True:
        try:
            _, loss, classification_loss, weight_loss, \
            tower_logits, tower_labels, = sess.run([ops['train_op'],
                                                    ops['loss'],
                                                    ops['classification_loss'],
                                                    ops['weight_loss'],
                                                    ops['tower_logits'],
                                                    ops['tower_labels']],
                                                   feed_dict=feed_dict)
            for logits, labels in zip(tower_logits, tower_labels):
                pred = np.argmax(logits, -1)
                correct = np.mean(pred == labels)
                acc_meter.update(correct, pred.shape[0])

            # update meters
            loss_meter.update(loss)
            cls_loss_meter.update(classification_loss)
            weight_loss_meter.update(weight_loss)
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % config.print_freq == 0:
                logger.info(f'Train: [{epoch}][{batch_idx}] '
                            f'T {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            f'acc {acc_meter.val:.3f} ({acc_meter.avg:.3f}) '
                            f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) '
                            f'cls loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f}) '
                            f'weight loss {weight_loss_meter.val:.3f} ({weight_loss_meter.avg:.3f})')
            batch_idx += 1
        except tf.errors.OutOfRangeError:
            break


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
    _, config = parse_option()
    os.makedirs(config.log_dir, exist_ok=True)
    logger = setup_logger(output=config.log_dir, name="modelnet40")
    logger.info(pprint.pformat(config))
    save_path = training(config)
    evaluating(config, save_path, config.gpus[0])
