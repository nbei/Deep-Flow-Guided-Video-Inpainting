import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import argparse
import yaml

import torch
import cvbase as cvb
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.loss_func as L
from dataset.FlowRefine import FlowSeq
from models import resnet_models
from utils.io import save_ckpt, load_ckpt
from utils.runner_func import *


parser = argparse.ArgumentParser()

# training options
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--model_name', type=str, default=None)

parser.add_argument('--max_iter', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=32)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--get_mask', action='store_true')

parser.add_argument('--LR', type=float, default=5e-5)
parser.add_argument('--LAMBDA_SMOOTH', type=float, default=0.1)
parser.add_argument('--LAMBDA_HARD', type=float, default=1.)
parser.add_argument('--BETA1', type=float, default=0.9)
parser.add_argument('--BETA2', type=float, default=0.999)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.00004)

parser.add_argument('--IMAGE_SHAPE', type=int, default=[320, 600], nargs='+')
parser.add_argument('--RES_SHAPE', type=int, default=[320, 600], nargs='+')
parser.add_argument('--FIX_MASK', action='store_true')
parser.add_argument('--MASK_MODE', type=str, default=None)
parser.add_argument('--PRETRAINED', action='store_true')
parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--RESNET_PRETRAIN_MODEL', type=str, default='./pretrained_models/resnet50-19c8e357.pth')
parser.add_argument('--TRAIN_LIST', type=str, default=None)
parser.add_argument('--MASK_ROOT', type=str, default=None)
parser.add_argument('--DATA_ROOT', type=str, default=None,
                    help='Set the path to flow dataset')
parser.add_argument('--GT_FLOW_ROOT', type=str, default=None)

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--SAMPLE_STEP', type=int, default=1000)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=10000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=10000)
parser.add_argument('--CPU', action='store_true')

parser.add_argument('--MASK_HEIGHT', type=int, default=80)
parser.add_argument('--MASK_WIDTH', type=int, default=150)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=60)
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=106)

parser.add_argument('--lr_decay_steps', type=int, nargs='+',
                    default=[50000, 100000, 300000])

args = parser.parse_args()


def main():

    image_size = [args.IMAGE_SHAPE[0], args.IMAGE_SHAPE[1]]

    if args.model_name is not None:
        model_save_dir = './snapshots/'+args.model_name+'/ckpt/'
        sample_dir = './snapshots/'+args.model_name+'/images/'
        log_dir = './logs/'+args.model_name
    else:
        model_save_dir = os.path.join(args.save_dir, 'ckpt')
        sample_dir = os.path.join(args.save_dir, 'images')
        log_dir = args.log_dir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(7777777)
    if not args.CPU:
        torch.cuda.manual_seed(7777777)

    flow_resnet = resnet_models.Flow_Branch_Multi(input_chanels=66, NoLabels=4)
    saved_state_dict = torch.load(args.RESNET_PRETRAIN_MODEL)
    for i in saved_state_dict:
        if 'conv1.' in i[:7]:
            conv1_weight = saved_state_dict[i]
            conv1_weight_mean = torch.mean(conv1_weight, dim=1, keepdim=True)
            conv1_weight_new = (conv1_weight_mean / 66.0).repeat(1, 66, 1, 1)
            saved_state_dict[i] = conv1_weight_new
    flow_resnet.load_state_dict(saved_state_dict, strict=False)
    flow_resnet = nn.DataParallel(flow_resnet).cuda()
    flow_resnet.train()

    optimizer = optim.SGD([{'params': get_1x_lr_params(flow_resnet.module), 'lr': args.LR},
                           {'params': get_10x_lr_params(flow_resnet.module), 'lr': 10 * args.LR}],
                          lr=args.LR, momentum=0.9, weight_decay=args.WEIGHT_DECAY)

    train_dataset = FlowSeq(args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.n_threads)

    if args.resume:
        if args.PRETRAINED_MODEL is not None:
            resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                                    [('model', flow_resnet)],
                                    [('optimizer', optimizer)],
                                    strict=True)
            print('Model Resume from', resume_iter, 'iter')
        else:
            print('Cannot load Pretrained Model')
            return

    if args.PRETRAINED:
        if args.PRETRAINED_MODEL is not None:
            resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                                    [('model', flow_resnet)],
                                    strict=True)
            print('Model Resume from', resume_iter, 'iter')

    train_iterator = iter(train_loader)

    loss = {}

    start_iter = 0 if not args.resume else resume_iter

    for i in tqdm(range(start_iter, args.max_iter)):
        try:
            flow_mask_cat, flow_masked, gt_flow, mask = next(train_iterator)
        except:
            print('Loader Restart')
            train_iterator = iter(train_loader)
            flow_mask_cat, flow_masked, gt_flow, mask = next(train_iterator)

        input_x = flow_mask_cat.cuda()
        gt_flow = gt_flow.cuda()
        mask = mask.cuda()
        flow_masked = flow_masked.cuda()

        flow1x = flow_resnet(input_x)
        f_res = flow1x[:, :2, :, :]
        r_res = flow1x[:, 2:, :, :]

        # fake_flow_f = f_res * mask[:,10:12,:,:] + flow_masked[:,10:12,:,:] * (1. - mask[:,10:12,:,:])
        # fake_flow_r = r_res * mask[:,32:34,:,:] + flow_masked[:,32:34,:,:] * (1. - mask[:,32:34,:,:])

        loss['1x_recon'] = L.L1_mask(f_res, gt_flow[:,:2,:,:], mask[:,10:12,:,:])
        loss['1x_recon'] += L.L1_mask(r_res, gt_flow[:, 2:, ...], mask[:, 32:34, ...])
        loss['f_recon_hard'], new_mask = L.L1_mask_hard_mining(f_res, gt_flow[:, :2,:,:], mask[:,10:11,:,:])
        loss['r_recon_hard'], new_mask = L.L1_mask_hard_mining(r_res, gt_flow[:, 2:, ...], mask[:, 32:33, ...])

        loss_total = loss['1x_recon'] + args.LAMBDA_HARD * (loss['f_recon_hard'] + loss['r_recon_hard'])

        if i % args.NUM_ITERS_DECAY == 0:
            adjust_learning_rate(optimizer, i, args.lr_decay_steps)
            print('LR has been changed')

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if i % args.PRINT_EVERY == 0:
            print('=========================================================')
            print(args.model_name, "Rank[{}] Iter [{}/{}]".format(0, i + 1, args.max_iter))
            print('=========================================================')
            print_loss_dict(loss)
            write_loss_dict(loss, writer, i)

        if (i+1) % args.MODEL_SAVE_STEP == 0:
            save_ckpt(os.path.join(model_save_dir, 'DFI_%d.pth' % i),
                      [('model', flow_resnet)], [('optimizer', optimizer)], i)
            print('Model has been saved at %d Iters' % i)

    writer.close()


if __name__ == '__main__':
    main()
