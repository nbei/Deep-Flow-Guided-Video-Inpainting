import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import torch
import numpy as np
import cv2

from models import DeepFill


class DeepFillv1(object):
    def __init__(self,
                 pretrained_model=None,
                 image_shape=[480, 840],
                 res_shape=None,
                 device=torch.device('cuda:0')):
        self.image_shape = image_shape
        self.res_shape = res_shape
        self.device = device

        self.deepfill = DeepFill.Generator().to(device)
        model_weight = torch.load(pretrained_model)
        self.deepfill.load_state_dict(model_weight, strict=True)
        self.deepfill.eval()
        print('Load Deepfill Model from', pretrained_model)

    def forward(self, img, mask):

        img, mask, small_mask = self.data_preprocess(img, mask, size=self.image_shape)

        image = torch.stack([img])
        mask = torch.stack([mask])
        small_mask = torch.stack([small_mask])

        with torch.no_grad():
            _, inpaint_res, _ = self.deepfill(image.to(self.device), mask.to(self.device), small_mask.to(self.device))

        res_complete = self.data_proprocess(image, mask, inpaint_res)

        return res_complete

    def data_preprocess(self, img, mask, enlarge_kernel=0, size=[480, 840]):
        img = img / 127.5 - 1
        mask = (mask > 0).astype(np.int)
        img = cv2.resize(img, (size[1], size[0]))
        if enlarge_kernel > 0:
            kernel = np.ones((enlarge_kernel, enlarge_kernel), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = (mask > 0).astype(np.uint8)

        small_mask = cv2.resize(mask, (size[1] // 8, size[0] // 8), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1]
        else:
            mask = np.expand_dims(mask, axis=2)

        if len(small_mask.shape) == 3:
            small_mask = small_mask[:, :, 0:1]
        else:
            small_mask = np.expand_dims(small_mask, axis=2)

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous().float()
        small_mask = torch.from_numpy(small_mask).permute(2, 0, 1).contiguous().float()

        return img*(1-mask), mask, small_mask

    def data_proprocess(self, img, mask, res):
        img = img.cpu().data.numpy()[0]
        mask = mask.data.numpy()[0]
        res = res.cpu().data.numpy()[0]

        res_complete = res * mask + img * (1. - mask)
        res_complete = (res_complete + 1) * 127.5
        res_complete = res_complete.transpose(1, 2, 0)
        if self.res_shape is not None:
            res_complete = cv2.resize(res_complete,
                                      (self.res_shape[1], self.res_shape[0]))

        return res_complete


def parse_arges():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_shape', type=int, nargs='+',
                        default=[480, 840])
    parser.add_argument('--res_shape', type=int, nargs='+',
                        default=None)
    parser.add_argument('--pretrained_model', type=str,
                        default='./pretrained_models/imagenet_deepfill.pth')
    parser.add_argument('--test_img', type=str,
                        default='./00000.jpg')
    parser.add_argument('--test_mask', type=str,
                        default='./00000.png')
    parser.add_argument('--output_path', type=str,
                        default='./res_00000.png')

    args = parser.parse_args()

    return args


def main():

    args = parse_arges()

    deepfill = DeepFillv1(pretrained_model=args.pretrained_model,
                          image_shape=args.image_shape,
                          res_shape=args.res_shape)

    test_image = cv2.imread(args.test_img)
    mask = cv2.imread(args.test_mask, cv2.IMREAD_UNCHANGED)

    with torch.no_grad():
        img_res = deepfill.forward(test_image, mask)

    cv2.imwrite(args.output_path, img_res)
    print('Result Saved')


if __name__ == '__main__':
    main()