import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import cv2
import torch.nn.functional as F



def random_bbox(args):
    img_shape = args.IMAGE_SHAPE
    img_height = img_shape[0]
    img_width = img_shape[1]

    maxt = img_height - args.VERTICAL_MARGIN - args.MASK_HEIGHT
    maxl = img_width - args.HORIZONTAL_MARGIN - args.MASK_WIDTH

    t = randint(args.VERTICAL_MARGIN, maxt)
    l = randint(args.HORIZONTAL_MARGIN, maxl)
    h = args.MASK_HEIGHT
    w = args.MASK_WIDTH
    return (t, l, h, w)


def mid_bbox_mask(args):

    def npmask(bbox, height, width):
        mask = np.zeros((1, 1, height, width), np.float32)
        mask[:, :, bbox[0]: bbox[0] + bbox[2],
        bbox[1]: bbox[1] + bbox[3]] = 1.
        return mask
    img_shape = args.IMAGE_SHAPE
    height = img_shape[0]
    width = img_shape[1]
    bbox = (height * 3 // 8, width * 3 // 8, args.MASK_HEIGHT, args.MASK_WIDTH)
    mask = npmask(bbox, height, width)

    return mask


def bbox2mask(args, bbox):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [B, 1, H, W]

    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, 1, height, width), np.float32)
        h = np.random.randint(delta_h // 2 + 1)
        w = np.random.randint(delta_w // 2 + 1)
        mask[:, :, bbox[0] + h: bbox[0] + bbox[2] - h,
        bbox[1] + w: bbox[1] + bbox[3] - w] = 1.
        return mask

    img_shape = args.IMAGE_SHAPE
    height = img_shape[0]
    width = img_shape[1]

    mask = npmask(bbox, height, width,
                  args.MAX_DELTA_HEIGHT,
                  args.MAX_DELTA_WIDTH)
    # small_mask = cv2.resize(mask[0].transpose(1, 2, 0), (width//8, height//8),
    #                         interpolation=cv2.INTER_NEAREST)
    # if len(small_mask.shape) < 3:
    #     small_mask = np.expand_dims(small_mask, 2)
    # small_mask = small_mask.transpose(2, 0, 1)
    # small_mask = np.expand_dims(small_mask, axis=0)

    return mask


def bbox2mask_background(args, bbox, back_mask):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [B, 1, H, W]

    """

    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, 1, height, width), np.float32)
        h = np.random.randint(delta_h // 2 + 1)
        w = np.random.randint(delta_w // 2 + 1)
        mask[:, :, bbox[0] + h: bbox[0] + bbox[2] - h,
        bbox[1] + w: bbox[1] + bbox[3] - w] = 1.
        return mask

    img_shape = args.IMG_SHAPE
    height = img_shape[0]
    width = img_shape[1]

    mask = npmask(bbox, height, width,
                  args.MAX_DELTA_HEIGHT,
                  args.MAX_DELTA_WIDTH)

    return torch.FloatTensor(mask) * back_mask


def ff_mask(args):
    config_mask = {'MAXVERTEX': 6,
                   'MAXANGLE': 6.0,
                   'MAXLENGTH': 30,
                   'MAXBRUSHWIDTH': 10}
    h = args.IMG_SHAPE[0]
    w = args.IMG_SHAPE[1]
    c = 3

    mask = random_ff_mask((h, w, c), config_mask)
    small_mask = cv2.resize(mask, (w // 8, h // 8),
                            interpolation=cv2.INTER_NEAREST)

    if len(small_mask.shape) < 3:
        small_mask = np.expand_dims(small_mask, 2)
    small_mask = small_mask.transpose(2, 0, 1)
    small_mask = np.expand_dims(small_mask, axis=0)

    mask = np.transpose(mask, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)

    return torch.FloatTensor(mask), torch.FloatTensor(small_mask)


def ff_mask_backgroud(args, back_mask):

    config_mask = {'MAXVERTEX': 8,
                    'MAXANGLE': 4.0,
                    'MAXLENGTH': 40,
                    'MAXBRUSHWIDTH': 4}
    h = args.IMAGE_SHAPE[0]
    w = args.IMAGE_SHAPE[1]
    c = 3

    mask = random_ff_mask((h, w, c), config_mask)

    mask = np.transpose(mask, (2, 0, 1))
    mask = np.expand_dims(mask, axis=0)

    return torch.FloatTensor(mask) * back_mask

def random_ff_mask(img_shape, config):
    """

    :param img_shape: (h, w, c)
    :param config:  'MAXVERTEX': 8,
                   'MAXANGLE': 4.0,
                   'MAXLENGTH': 40,
                   'MAXBRUSHWIDTH': 4
    :return: mask with shape (h, w, c) , value from 0-1 and the hole
             of the mask is filled by 1
    """
    h, w, c = img_shape

    def npmaask():

        mask = np.zeros((h, w))
        num_v = 4 + np.random.randint(config['MAXVERTEX'])

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(config['MAXANGLE'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(config['MAXLENGTH'])
                brush_w = 10 + np.random.randint(config['MAXBRUSHWIDTH'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape(mask.shape + (1,)).astype(np.float32)

    mask = npmaask()
    return mask
