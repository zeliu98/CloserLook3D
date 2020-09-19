"""
Distributed evaluating script for 3D shape classification with ModelNet40 dataset
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
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from models import build_classification
from datasets import ModelNet40Cls
import datasets.data_utils as d_utils
from utils.util import AverageMeter, accuracy
from utils.logger import setup_logger
from utils.config import config, update_config


def parse_option():
    parser = argparse.ArgumentParser('Modelnet40 classification evaluating')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--load_path', required=True, type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--log_dir', type=str, default='log_eval', help='log dir [default: log_eval]')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    args, unparsed = parser.parse_known_args()

    update_config(args.cfg)

    config.data_root = args.data_root
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.rng_seed = args.rng_seed
    config.local_rank = args.local_rank

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    config.log_dir = os.path.join(args.log_dir, 'modelnet40', f'{ddir_name}_{int(time.time())}')

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points

    print(args)
    print(config)

    # torch.manual_seed(args.rng_seed)
    # torch.cuda.manual_seed_all(args.rng_seed)
    # random.seed(args.rng_seed)
    # np.random.seed(args.rng_seed)

    return args, config


def get_loader(args):
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    test_dataset = ModelNet40Cls(input_features_dim=config.input_features_dim, num_points=args.num_points,
                                 data_root=args.data_root, transforms=test_transforms,
                                 subsampling_parameter=config.sampleDl,
                                 split='test')

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)

    return test_loader


def load_checkpoint(config, model):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def main(config):
    test_loader = get_loader(config)
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model, criterion = build_classification(config)
    model.cuda()
    criterion.cuda()

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model)
        logger.info("==> checking loaded ckpt")
        validate(test_loader, model, criterion, config, num_votes=10)


def validate(test_loader, model, criterion, config, num_votes=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        vote_preds = None
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low,
                                                   scale_high=config.scale_high,
                                                   std=config.noise_std,
                                                   clip=config.noise_clip)
        for v in range(num_votes):
            preds = []
            targets = []
            for idx, (points, mask, features, target) in enumerate(test_loader):
                # augment for voting
                if v > 0:
                    points = TS(points)
                    if config.input_features_dim == 3:
                        features = points
                        features = features.transpose(1, 2).contiguous()
                    elif config.input_features_dim == 4:
                        features = torch.ones(size=(points.shape[0], points.shape[1], 1), dtype=torch.float32)
                        features = torch.cat([features, points], -1)
                        features = features.transpose(1, 2).contiguous()
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.input_features_dim} in voting not supported")

                # forward
                points = points.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                features = features.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                pred = model(points, mask, features)
                target = target.view(-1)
                loss = criterion(pred, target)
                acc1 = accuracy(pred, target, topk=(1,))
                losses.update(loss.item(), points.size(0))
                top1.update(acc1[0].item(), points.size(0))

                preds.append(pred)
                targets.append(target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % config.print_freq == 0:
                    logger.info(
                        f'Test: [{idx}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})')
            logger.info(f' * Acc@1 {top1.avg:.3%}')
            top1.reset()

            preds = torch.cat(preds, 0)
            targets = torch.cat(targets, 0)
            if vote_preds is None:
                vote_preds = preds
            else:
                vote_preds += preds
            vote_acc1 = accuracy(vote_preds, targets, topk=(1,))[0].item()
            logger.info(f' * Vote{v} Acc@1 {vote_acc1:.3%}')

    return vote_acc1


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.makedirs(opt.log_dir, exist_ok=True)

    os.environ["JOB_LOAD_DIR"] = os.path.dirname(config.load_path)

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="modelnet40_eval")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))
    main(config)
