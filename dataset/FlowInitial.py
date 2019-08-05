import torch
import os
import random
import cv2
import cvbase as cvb
import numpy as np
import torch.utils.data as data
import utils.image as im
import utils.region_fill as rf


class FlowSeq(data.Dataset):

    def __init__(self, config, isTest=False):
        super(FlowSeq, self).__init__()
        self.config = config
        self.data_items = []
        self.size = self.config.IMAGE_SHAPE
        self.res_size = self.config.RES_SHAPE
        self.isTest = isTest
        self.data_list = config.EVAL_LIST if isTest else config.TRAIN_LIST
        with open(self.data_list, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.strip(' ')
                line_split = line.split(' ')

                flow_dir = line_split[0:11]
                if self.config.DATA_ROOT is not None:
                    flow_dir = [os.path.join(self.config.DATA_ROOT, x) for x in flow_dir]
                if self.config.get_mask:
                    mask_dir = line_split[11:22]
                    if not self.config.FIX_MASK:
                        mask_dir = [os.path.join(self.config.MASK_ROOT, x) for x in mask_dir]
                    else:
                        mask_dir = [os.path.join(self.config.MASK_ROOT) for x in mask_dir]
                video_class_no = int(line_split[-1])
                if not self.isTest:
                    self.data_items.append((flow_dir, video_class_no))
                else:
                    output_dirs = line_split[-2]
                    if self.config.get_mask:
                        self.data_items.append((flow_dir, video_class_no, mask_dir, output_dirs))
                    else:
                        self.data_items.append((flow_dir, video_class_no, output_dirs))

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):

        flow_dir = self.data_items[idx][0]
        video_class_no = self.data_items[idx][1]
        if self.config.get_mask:
            mask_dir = self.data_items[idx][2]
        if self.isTest:
            output_dirs = self.data_items[idx][-1]
        flow_set = []
        mask_set = []
        flow_mask_cat_set = []
        flow_masked_set = []

        if self.config.MASK_MODE == 'bbox':
            tmp_bbox = im.random_bbox(self.config)
            tmp_mask = im.bbox2mask(self.config, tmp_bbox)
            tmp_mask = tmp_mask[0, 0, :, :]
            fix_mask = np.expand_dims(tmp_mask, axis=2)
        elif self.config.MASK_MODE == 'mid-bbox':
            tmp_mask = im.mid_bbox_mask(self.config)
            tmp_mask = tmp_mask[0, 0, :, :]
            fix_mask = np.expand_dims(tmp_mask, axis=2)

        for i in range(11):
            tmp_flow = cvb.read_flow(flow_dir[i])
            if self.config.get_mask:
                tmp_mask = cv2.imread(mask_dir[i],
                                      cv2.IMREAD_UNCHANGED)
                tmp_mask = self._mask_tf(tmp_mask)
            else:
                if self.config.FIX_MASK:
                    tmp_mask = fix_mask.copy()
                else:
                    tmp_bbox = im.random_bbox(self.config)
                    tmp_mask = im.bbox2mask(self.config, tmp_bbox)
                    tmp_mask = tmp_mask[0, 0, :, :]
                    tmp_mask = np.expand_dims(tmp_mask, axis=2)
            tmp_flow = self._flow_tf(tmp_flow)
            tmp_flow_masked = tmp_flow * (1. - tmp_mask)

            if self.config.INITIAL_HOLE:
                tmp_flow_resized = cv2.resize(tmp_flow, (self.size[1] // 2, self.size[0] // 2))
                tmp_mask_resized = cv2.resize(tmp_mask, (self.size[1] // 2, self.size[0] // 2), cv2.INTER_NEAREST)
                tmp_flow_masked_small = tmp_flow_resized
                tmp_flow_masked_small[:, :, 0] = rf.regionfill(tmp_flow_resized[:, :, 0], tmp_mask_resized)
                tmp_flow_masked_small[:, :, 1] = rf.regionfill(tmp_flow_resized[:, :, 1], tmp_mask_resized)

                tmp_flow_masked = tmp_flow_masked + \
                                  tmp_mask * cv2.resize(tmp_flow_masked_small, (self.size[1], self.size[0]))

            flow_masked_set.append(tmp_flow_masked)
            flow_set.append(tmp_flow)
            mask_set.append(tmp_mask)
            mask_set.append(tmp_mask)
            tmp_flow_mask_cat = np.concatenate((tmp_flow_masked, tmp_mask), axis=2)
            flow_mask_cat_set.append(tmp_flow_mask_cat)

        flow_mask_cat = np.concatenate(flow_mask_cat_set, axis=2)
        flow_masked = np.concatenate(flow_masked_set, axis=2)
        gt_flow = np.concatenate(flow_set, axis=2)
        mask = np.concatenate(mask_set, axis=2)

        flow_mask_cat = torch.from_numpy(flow_mask_cat).permute(2, 0, 1).contiguous().float()
        flow_masked = torch.from_numpy(flow_masked).permute(2, 0, 1).contiguous().float()
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous().float()

        if self.isTest:
            return flow_mask_cat, flow_masked, gt_flow, mask, output_dirs

        return flow_mask_cat, flow_masked, gt_flow, mask

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))
        img = img / 127.5 - 1

        return img

    def _mask_tf(self, mask):
        mask = cv2.resize(mask, (self.size[1], self.size[0]),
                          interpolation=cv2.INTER_NEAREST)
        if self.config.enlarge_mask:
            enlarge_kernel = np.ones((self.config.enlarge_kernel, self.config.enlarge_kernel),
                                     np.uint8)
            tmp_mask = cv2.dilate(mask[:, :, 0], enlarge_kernel, iterations=1)
            mask[(tmp_mask > 0), :] = 255

        mask = mask[:,:,0]
        mask = np.expand_dims(mask, axis=2)
        mask = mask / 255

        return mask

    def _flow_tf(self, flow):
        origin_shape = flow.shape
        flow = cv2.resize(flow, (self.res_size[1], self.res_size[0]))
        flow[:, :, 0] = flow[:, :, 0].clip(-1. * origin_shape[1], origin_shape[1]) / origin_shape[1] * self.res_size[1]
        flow[:, :, 1] = flow[:, :, 1].clip(-1. * origin_shape[0], origin_shape[0]) / origin_shape[0] * self.res_size[0]

        return flow
