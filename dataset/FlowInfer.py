import torch
import cv2
import numpy as np
import torch.utils.data


class FlowInfer(torch.utils.data.Dataset):

    def __init__(self, list_file, size=None, isRGB=True, start_pos=0):
        super(FlowInfer, self).__init__()
        self.size = size
        txt_file = open(list_file, 'r')
        self.frame1_list = []
        self.frame2_list = []
        self.output_list = []
        self.isRGB = isRGB

        for line in txt_file:
            line = line.strip(' ')
            line = line.strip('\n')

            line_split = line.split(' ')
            self.frame1_list.append(line_split[0])
            self.frame2_list.append(line_split[1])
            self.output_list.append(line_split[2])

        if start_pos > 0:
            self.frame1_list = self.frame1_list[start_pos:]
            self.frame2_list = self.frame2_list[start_pos:]
            self.output_list = self.output_list[start_pos:]
        txt_file.close()

    def __len__(self):
        return len(self.frame1_list)

    def __getitem__(self, idx):
        frame1 = cv2.imread(self.frame1_list[idx])
        frame2 = cv2.imread(self.frame2_list[idx])
        if self.isRGB:
            frame1 = frame1[:, :, ::-1]
            frame2 = frame2[:, :, ::-1]
        output_path = self.output_list[idx]

        frame1 = self._img_tf(frame1)
        frame2 = self._img_tf(frame2)

        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).contiguous().float()
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).contiguous().float()

        return frame1_tensor, frame2_tensor, output_path

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))

        return img
