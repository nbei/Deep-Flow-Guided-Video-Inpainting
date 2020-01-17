from dataset.FlowInfer import FlowInfer
from models import LiteFlowNet
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
import cvbase as cvb
import torch
import numpy as np
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model_liteflownet', type=str,
                        default='./pretrained_models/default.pth')
    parser.add_argument('--img_size', type=list, default=(512, 1024, 3))
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--data_list', type=str, default=None, help='Give the data list to extract flow')
    parser.add_argument('--frame_dir', type=str, default=None,
                        help='Give the dir of the video frames and generate the data list to extract flow')

    args = parser.parse_args()
    return args


def estimate(model, tensorFirst, tensorSecond):
    # print(tensorFirst.shape)
    #assert(tensorFirst.size(1) == tensorSecond.size(1))
    #assert(tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(3)
    intHeight = tensorFirst.size(2)

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    #tensorPreprocessedFirst = tensorFirst.to(device).view(1, 3, intHeight, intWidth)
    #tensorPreprocessedSecond = tensorSecond.to(device).view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(1024 / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(436 / 32.0) * 32.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tensorFlow = torch.nn.functional.interpolate(input=model(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tensorFlow


def infer(args):
    assert args.data_list is not None or args.frame_dir is not None

    if args.frame_dir is not None:
        data_list = generate_flow_list(args.frame_dir)
        args.data_list = data_list

    device = torch.device('cuda:0')

    print('====> Loading', args.pretrained_model_liteflownet)
    Flownet = LiteFlowNet(args.pretrained_model_liteflownet)
    Flownet.to(device)
    Flownet.eval()

    dataset_ = FlowInfer(args.data_list, size=args.img_size)
    dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)
    #task_bar = ProgressBar(dataset_.__len__())
    with torch.no_grad():
        for i, (f1, f2, output_path_) in tqdm(enumerate(dataloader_), total=len(dataset_)):
            f1 = f1.to(device)
            f2 = f2.to(device)

            flow = estimate(Flownet, f1, f2)

            output_path = output_path_[0]

            output_file = os.path.dirname(output_path)
            if not os.path.exists(output_file):
                os.makedirs(output_file)

            flow_numpy = flow[0].permute(1, 2, 0).data.cpu().numpy()
            cvb.write_flow(flow_numpy, output_path)

    sys.stdout.write('\n')
    print('LiteFlowNet Inference has been finished~!')
    print('Extracted Flow has been save in', output_file)

    return output_file


def generate_flow_list(frame_dir):
    dataset_root = os.path.dirname(frame_dir)
    video_root = frame_dir
    train_list = open(os.path.join(dataset_root, 'video.txt'), 'w')
    flow_list = open(os.path.join(dataset_root, 'video_flow.txt'), 'w')
    output_root = os.path.join(dataset_root, 'Flow')

    img_total = 0
    video_id = os.path.basename(frame_dir)

    img_id_list = [x for x in os.listdir(video_root) if '.png' in x or '.jpg' in x]
    img_id_list.sort()
    img_num = len(img_id_list)
    train_list.write(video_id)
    train_list.write(' ')
    train_list.write(str(img_num))
    train_list.write('\n')
    img_total += img_num

    for i in range(img_num):
        if i + 1 < img_num:
            flow_list.write(os.path.join(video_root, img_id_list[i]))
            flow_list.write(' ')
            flow_list.write(os.path.join(video_root, img_id_list[i + 1]))
            flow_list.write(' ')
            flow_list.write(os.path.join(output_root, img_id_list[i][:-4] + '.flo'))
            flow_list.write('\n')

        if i - 1 >= 0:
            flow_list.write(os.path.join(video_root, img_id_list[i]))
            flow_list.write(' ')
            flow_list.write(os.path.join(video_root, img_id_list[i - 1]))
            flow_list.write(' ')
            flow_list.write(os.path.join(output_root, img_id_list[i][:-4] + '.rflo'))
            flow_list.write('\n')

    print('This Video has', img_total, 'Images')
    train_list.close()
    flow_list.close()
    print('The optical flow list has been generated:',
          os.path.join(dataset_root, 'video_flow.txt'))

    return os.path.join(dataset_root, 'video_flow.txt')


def main():
    args = parse_args()
    infer(args)


if __name__ == '__main__':
    main()
