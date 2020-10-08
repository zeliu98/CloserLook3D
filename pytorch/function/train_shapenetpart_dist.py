"""
Distributed training script for part segmentation with ShapeNetPart dataset
"""
import argparse
import os
import sys
import time
import json
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

import datasets.data_utils as d_utils
from models import build_multi_part_segmentation
from datasets import ShapeNetPartSeg
from utils.util import AverageMeter, shapenetpart_metrics
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from utils.config import config, update_config


def parse_option():
    parser = argparse.ArgumentParser('ShapeNetPart part-segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--data_root', type=str, default='data', metavar='PATH', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight_decay')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, help='used for resume')

    # io
    parser.add_argument('--load_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')
    parser.add_argument('--log_dir', type=str, default='log', help='log dir [default: log]')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.print_freq = args.print_freq
    config.save_freq = args.save_freq
    config.val_freq = args.val_freq
    config.rng_seed = args.rng_seed

    config.local_rank = args.local_rank

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'shapenetpart', f'{ddir_name}_{int(time.time())}')

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.base_learning_rate:
        config.base_learning_rate = args.base_learning_rate
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.epochs:
        config.epochs = args.epochs
    if args.start_epoch:
        config.start_epoch = args.start_epoch

    print(args)
    print(config)

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    return args, config


def get_loader(args):
    # set the data loader
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudScale(scale_low=config.scale_low, scale_high=config.scale_high),
        d_utils.PointcloudJitter(std=config.noise_std, clip=config.noise_clip),
    ])

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    train_dataset = ShapeNetPartSeg(num_points=args.num_points,
                                    data_root=args.data_root, transforms=train_transforms,
                                    split='trainval')
    test_dataset = ShapeNetPartSeg(num_points=args.num_points,
                                   data_root=args.data_root, transforms=test_transforms,
                                   split='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)

    return train_loader, test_loader


def load_checkpoint(config, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(config.log_dir, 'current.pth'))
    if epoch % config.save_freq == 0:
        torch.save(state, os.path.join(config.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, f'ckpt_epoch_{epoch}.pth')))


def main(config):
    train_loader, test_loader = get_loader(config)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model, criterion = build_multi_part_segmentation(config)
    logger.info(str(model))
    model.cuda()
    criterion.cuda()

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.batch_size * dist.get_world_size() / 16 * config.base_learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.base_learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.base_learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported")

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model, optimizer, scheduler)
        logger.info("==> checking loaded ckpt")
        validate('resume', 'test', test_loader, model, criterion, config)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=config.log_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(config.start_epoch, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config)

        logger.info('epoch {}, total time {:.2f}, lr {:.5f}'.format(epoch,
                                                                    (time.time() - tic),
                                                                    optimizer.param_groups[0]['lr']))
        acc, msIoU, mIoU = validate(epoch, test_loader, model, criterion, config, num_votes=1)

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(config, epoch, model, optimizer, scheduler)

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('test_acc', acc, epoch)
            summary_writer.add_scalar('msIoU', msIoU, epoch)
            summary_writer.add_scalar('mIoU', mIoU, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


def train(epoch, train_loader, model, criterion, optimizer, scheduler, config):
    """
    One epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()
    for idx, (points, mask, points_labels, shape_labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = points.size(0)
        # forward
        features = points
        features = features.transpose(1, 2).contiguous()

        points = points.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        features = features.cuda(non_blocking=True)
        points_labels = points_labels.cuda(non_blocking=True)
        shape_labels = shape_labels.cuda(non_blocking=True)

        pred = model(points, mask, features)
        loss = criterion(pred, points_labels, shape_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % config.print_freq == 0:
            logger.info(f'Train: [{epoch}/{config.epochs + 1}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')
    return loss_meter.avg


def validate(epoch, test_loader, model, criterion, config, num_votes=10):
    """
    One epoch validating
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        all_logits = []
        all_points_labels = []
        all_shape_labels = []
        all_masks = []
        end = time.time()
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low,
                                                   scale_high=config.scale_high,
                                                   std=config.noise_std,
                                                   clip=config.noise_clip)

        for idx, (points_orig, mask, points_labels, shape_labels) in enumerate(test_loader):
            vote_logits = None
            vote_points_labels = None
            vote_shape_labels = None
            vote_masks = None
            for v in range(num_votes):
                batch_logits = []
                batch_points_labels = []
                batch_shape_labels = []
                batch_masks = []
                # augment for voting
                if v > 0:
                    points = TS(points_orig)
                else:
                    points = points_orig
                # forward
                features = points
                features = features.transpose(1, 2).contiguous()
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                points_labels = points_labels.cuda(non_blocking=True)
                shape_labels = shape_labels.cuda(non_blocking=True)

                pred = model(points, mask, features)
                loss = criterion(pred, points_labels, shape_labels)
                losses.update(loss.item(), points.size(0))

                # collect
                bsz = points.shape[0]
                for ib in range(bsz):
                    sl = shape_labels[ib]
                    logits = pred[sl][ib]
                    pl = points_labels[ib]
                    pmk = mask[ib]
                    batch_logits.append(logits.cpu().numpy())
                    batch_points_labels.append(pl.cpu().numpy())
                    batch_shape_labels.append(sl.cpu().numpy())
                    batch_masks.append(pmk.cpu().numpy().astype(np.bool))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if vote_logits is None:
                    vote_logits = batch_logits
                    vote_points_labels = batch_points_labels
                    vote_shape_labels = batch_shape_labels
                    vote_masks = batch_masks
                else:
                    for i in range(len(vote_logits)):
                        vote_logits[i] = vote_logits[i] + (batch_logits[i] - vote_logits[i]) / (v + 1)

            all_logits += vote_logits
            all_points_labels += vote_points_labels
            all_shape_labels += vote_shape_labels
            all_masks += vote_masks
            if idx % config.print_freq == 0:
                logger.info(
                    f'V{num_votes} Test: [{idx}/{len(test_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})')

        acc, shape_ious, msIoU, mIoU = shapenetpart_metrics(config.num_classes,
                                                            config.num_parts,
                                                            all_shape_labels,
                                                            all_logits,
                                                            all_points_labels,
                                                            all_masks)
        logger.info(f'E{epoch} V{num_votes} * mIoU {mIoU:.3%} msIoU {msIoU:.3%}')
        logger.info(f'E{epoch} V{num_votes} * Acc {acc:.3%}')
        logger.info(f'E{epoch} V{num_votes} * shape_ious {shape_ious}')

    return acc, msIoU, mIoU


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    os.makedirs(opt.log_dir, exist_ok=True)
    os.environ["JOB_LOG_DIR"] = config.log_dir

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="shapenetpart")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))
    main(config)
