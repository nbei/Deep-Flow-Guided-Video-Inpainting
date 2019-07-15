import torch.nn as nn
import torch
import torch.nn.functional as F

affine_par = True


def down_sample(x, scalor=2, mode='bilinear'):
    if mode == 'bilinear':
        x = F.avg_pool2d(x, kernel_size=scalor, stride=scalor)
    elif mode == 'nearest':
        x = F.max_pool2d(x, kernel_size=scalor, stride=scalor)

    return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,  # change
            padding=padding,
            bias=False,
            dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FlowBranch_Layer(nn.Module):
    def __init__(self, input_chanels, NoLabels):
        super(FlowBranch_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upconv1 = nn.Conv2d(input_chanels, input_chanels // 2, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.Conv2d(input_chanels // 2, 256, kernel_size=3, stride=1, padding=1)

        self.conv1_flow = nn.Conv2d(256, NoLabels, kernel_size=1, stride=1, padding=0)

        self.conv2_flow = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_flow = nn.Conv2d(128 + NoLabels, NoLabels, kernel_size=3, stride=1, padding=1)

        self.conv4_flow = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_flow = nn.Conv2d(64 + NoLabels, NoLabels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, input_size):
        x = self.upconv1(x)
        x = self.relu(x)
        x = F.upsample(x, (input_size[0] // 4, input_size[1] // 4), mode='bilinear', align_corners=False)
        x = self.upconv2(x)
        x = self.relu(x)

        res_4x = self.conv1_flow(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2_flow(x)
        x = self.relu(x)
        res_4x_up = F.upsample(res_4x, scale_factor=2, mode='bilinear', align_corners=False)
        conv3_input = torch.cat([x, res_4x_up], dim=1)
        res_2x = self.conv3_flow(conv3_input)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv4_flow(x)
        x = self.relu(x)
        res_2x_up = F.upsample(res_2x, scale_factor=2, mode='bilinear', align_corners=False)
        conv5_input = torch.cat([x, res_2x_up], dim=1)
        res_1x = self.conv5_flow(conv5_input)

        return res_1x, res_2x, res_4x


class FlowModule_MultiScale(nn.Module):
    def __init__(self, input_chanels, NoLabels):
        super(FlowModule_MultiScale, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(input_chanels, 256, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, NoLabels, kernel_size=1, padding=0)

    def forward(self, x, res_size):
        x = F.upsample(x, (res_size[0] // 4, res_size[1] // 4), mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.relu(x)
        x = F.upsample(x, (res_size[0] // 2, res_size[1] // 2), mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.upsample(x, res_size, mode='bilinear', align_corners=False)
        x = self.conv3(x)

        return x


class FlowModule_SingleScale(nn.Module):
    def __init__(self, input_channels, NoLabels):
        super(FlowModule_SingleScale, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, NoLabels, kernel_size=1, padding=0)

    def forward(self, x, res_size):
        x = self.conv1(x)
        x = F.upsample(x, res_size, mode='bilinear')

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, input_chanels, NoLabels, Layer5_Module=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_chanels, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=4)

        if Layer5_Module is not None:
            self.layer5 = Layer5_Module

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par), )
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        res = self.layer5(x, input_size)
        return res

    def train(self, mode=True, freezeBn=True):
        super(ResNet, self).train(mode=mode)
        if freezeBn:
            print("Freezing BatchNorm2D.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


def Flow_Branch(input_chanels=30, NoLabels=20):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   input_chanels, NoLabels,
                   Layer5_Module=FlowModule_SingleScale(2048, NoLabels))
    return model


def Flow_Branch_Multi(input_chanels=30, NoLabels=20, ResLabels=2048):
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_chanels, NoLabels, FlowModule_MultiScale(ResLabels, NoLabels))

    return model
