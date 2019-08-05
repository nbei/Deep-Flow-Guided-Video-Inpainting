import argparse, os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
import torch.nn as nn
import cv2
import numpy as np
import cvbase as cvb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmcv import ProgressBar

import utils.image as im
from models import resnet_models
from dataset import FlowInitial, FlowRefine
from utils.io import load_ckpt


def parse_args():
    parser = argparse.ArgumentParser()

    # training options
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=16)

    parser.add_argument('--get_mask', action='store_true')
    parser.add_argument('--output_root', type=str, default=None)

    parser.add_argument('--FIX_MASK', action='store_true')
    parser.add_argument('--DATA_ROOT', type=str,
                        default=None)
    parser.add_argument('--GT_FLOW_ROOT', type=str,
                        default=None)

    parser.add_argument('--MASK_MODE', type=str, default='bbox')
    parser.add_argument('--SAVE_FLOW', action='store_true')
    parser.add_argument('--MASK_ROOT', type=str, default=None)

    parser.add_argument('--IMAGE_SHAPE', type=int, default=[1024, 1024], nargs='+')
    parser.add_argument('--RES_SHAPE', type=int, default=[1024, 1024], nargs='+')
    parser.add_argument('--PRETRAINED', action='store_true')
    parser.add_argument('--ENLARGE_MASK', action='store_true')
    parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
    parser.add_argument('--INITIAL_HOLE', action='store_true')
    parser.add_argument('--EVAL_LIST', type=str, default=None)
    parser.add_argument('--PRINT_EVERY', type=int, default=10)

    parser.add_argument('--MASK_HEIGHT', type=int, default=120)
    parser.add_argument('--MASK_WIDTH', type=int, default=212)
    parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
    parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
    parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=30)
    parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=53)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.model_name == 'initial':
        test_initial_stage(args)
    elif args.model_name == 'refine':
        test_refine_stage(args)
    else:
        raise NotImplementedError('Please Choose correct testing mode')


def test_initial_stage(args):
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    args.INITIAL_HOLE = True
    args.get_mask = True

    eval_dataset = FlowInitial.FlowSeq(args, isTest=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.n_threads)

    if args.ResNet101:
        dfc_resnet101 = resnet_models.Flow_Branch(33, 2)
        dfc_resnet = nn.DataParallel(dfc_resnet101).cuda()
    else:
        dfc_resnet50 = resnet_models.Flow_Branch_Multi(input_chanels=33, NoLabels=2)
        dfc_resnet = nn.DataParallel(dfc_resnet50).cuda()

    dfc_resnet.eval()
    resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                            [('model', dfc_resnet)], strict=True)
    print('Load Pretrained Model from', args.PRETRAINED_MODEL)

    task_bar = ProgressBar(eval_dataset.__len__())
    for i, item in enumerate(eval_dataloader):
        with torch.no_grad():
            input_x = item[0].cuda()
            flow_masked = item[1].cuda()
            mask = item[3].cuda()
            output_dir = item[4][0]

            res_flow = dfc_resnet(input_x)
            res_complete = res_flow * mask[:, 10:11, :, :] + flow_masked[:, 10:12, :, :] * (1. - mask[:, 10:11, :, :])

            output_dir_split = output_dir.split(',')
            output_file = os.path.join(args.output_root, output_dir_split[0])
            output_basedir = os.path.dirname(output_file)
            if not os.path.exists(output_basedir):
                os.makedirs(output_basedir)
            res_save = res_complete[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            cvb.write_flow(res_save, output_file)
            task_bar.update()
    sys.stdout.write('\n')
    dfc_resnet = None
    torch.cuda.empty_cache()
    print('Initial Results Saved in', args.output_root)


def test_refine_stage(args):
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    eval_dataset = FlowRefine.FlowSeq(args, isTest=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.n_threads)

    if args.ResNet101:
        dfc_resnet101 = resnet_models.Flow_Branch(66, 4)
        dfc_resnet = nn.DataParallel(dfc_resnet101).cuda()
    else:
        dfc_resnet50 = resnet_models.Flow_Branch_Multi(input_chanels=66, NoLabels=4)
        dfc_resnet = nn.DataParallel(dfc_resnet50).cuda()

    dfc_resnet.eval()

    resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                            [('model', dfc_resnet)], strict=True)

    print('Load Pretrained Model from', args.PRETRAINED_MODEL)

    task_bar = ProgressBar(eval_dataset.__len__())
    for i, item in enumerate(eval_dataloader):
        with torch.no_grad():
            input_x = item[0].cuda()
            flow_masked = item[1].cuda()
            gt_flow = item[2].cuda()
            mask = item[3].cuda()
            output_dir = item[4][0]

            res_flow = dfc_resnet(input_x)

            res_flow_f = res_flow[:, :2, :, :]
            res_flow_r = res_flow[:, 2:, :, :]

            res_complete_f = res_flow_f * mask[:, 10:11, :, :] + flow_masked[:, 10:12, :, :] * (1. - mask[:, 10:11, :, :])
            res_complete_r = res_flow_r * mask[:,32:34,:,:] + flow_masked[:,32:34,:,:] * (1. - mask[:,32:34,:,:])

            output_dir_split = output_dir.split(',')

            output_file_f = os.path.join(args.output_root, output_dir_split[0])
            output_file_r = os.path.join(args.output_root, output_dir_split[1])
            output_basedir = os.path.dirname(output_file_f)
            if not os.path.exists(output_basedir):
                os.makedirs(output_basedir)

            res_save_f = res_complete_f[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            cvb.write_flow(res_save_f, output_file_f)
            res_save_r = res_complete_r[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            cvb.write_flow(res_save_r, output_file_r)
            task_bar.update()
    sys.stdout.write('\n')
    dfc_resnet = None
    torch.cuda.empty_cache()
    print('Refined Results Saved in', args.output_root)


if __name__ == '__main__':
    main()