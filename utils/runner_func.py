import numpy as np
import torch.nn.functional as F


# get resnet's params
def get_1x_lr_params(model):
    b = []
    b.append(model.conv1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.layer5)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


# helpful funcs in training process
def print_loss_dict(loss):

    for key, value in loss.items():
        print(key, ': ', value.cpu().data.numpy())


def write_loss_dict(loss, writer, iter):

    for key, value in loss.items():
        writer.add_scalar(key, value.cpu().data.numpy(), iter)


def adjust_learning_rate(optimizer, iter, iter_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1**(sum(iter >= np.array(iter_step)) - sum((iter-1) >= np.array(iter_step)))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def down_sample(x, scalor=2, mode='bilinear'):
    if mode == 'bilinear':
        x = F.avg_pool2d(x, kernel_size=scalor, stride=scalor)
    elif mode == 'nearest':
        x = F.max_pool2d(x, kernel_size=scalor, stride=scalor)

    return x