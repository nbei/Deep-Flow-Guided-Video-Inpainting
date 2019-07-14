import torch
from torch import nn
import numpy as np
from torch import autograd


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def TVLoss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def L1(x, y, mask=None):
    res = torch.abs(x - y)
    if mask is not None:
        res = res * mask
    return torch.mean(res)


def L1_mask(x, y, mask=None):
    res = torch.abs(x - y)
    if mask is not None:
        res = res * mask
        return torch.sum(res) / torch.sum(mask)
    return torch.mean(res)


def L1_mask_hard_mining(x, y, mask):
    input_size = x.size()
    res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
    with torch.no_grad():
        idx = mask > 0.5
        res_sort = [torch.sort(res[i, idx[i, ...]])[0] for i in range(idx.shape[0])]
        res_sort = [i[int(i.shape[0] * 0.5)].item() for i in res_sort]
        new_mask = mask.clone()
        for i in range(res.shape[0]):
            new_mask[i, ...] = ((mask[i, ...] > 0.5) & (res[i, ...] > res_sort[i])).float()

    res = res * new_mask
    final_res = torch.sum(res) / torch.sum(new_mask)
    return final_res, new_mask


def Boundary_Smoothness(x, mask):
    boundary_x = torch.abs(mask[:,:,1:,:] - mask[:,:,:-1,:])
    boundary_y = torch.abs(mask[:,:,:,1:] - mask[:,:,:,:-1])

    grad_x = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]), dim=1, keepdim=True)
    grad_y = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]), dim=1, keepdim=True)

    smoothness = torch.sum(grad_x * boundary_x) / torch.sum(boundary_x) + \
                 torch.sum(grad_y * boundary_y) / torch.sum(boundary_y)

    return smoothness


def Residual_Norm(residual):
    res = torch.sum(torch.abs(residual), dim=1)
    return torch.mean(res)


def get_gradient_x(img):
    grad_x = img[:,:,1:,:] - img[:,:,:-1,:]

    return grad_x

def get_gradient_y(img):
    grad_y = img[:,:,:,1:] - img[:,:,:,:-1]

    return grad_y


def get_flow_smoothness(fake_flow, true_flow):
    fake_grad_x = get_gradient_x(fake_flow)
    fake_grad_y = get_gradient_y(fake_flow)

    true_grad_x = get_gradient_x(true_flow)
    true_grad_y = get_gradient_y(true_flow)

    weight_x = torch.exp(-torch.mean(torch.abs(true_grad_x), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(true_grad_y), dim=1, keepdim=True))

    smoothness = torch.mean(torch.abs(fake_grad_x) * weight_x) + torch.mean(torch.abs(fake_grad_y) * weight_y)

    return smoothness

