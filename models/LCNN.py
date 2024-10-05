import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch import Tensor
import os
import numpy as np
import sys
from collections import OrderedDict


## Adapted from https://github.com/joaomonteirof/e2e_antispoofing
## https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/newfunctions/


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5 * torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)

            representations = torch.cat((avg_repr, std_repr), 1)

            return representations


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }


class ResNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(7, 3), stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        print(x.shape)
        x = self.activation(self.bn1(x))
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)

        x = self.activation(self.bn5(x)).squeeze(2)

        # x = self.activation(self.bn5(x))
        # x = self.conv6(x)
        # x = self.activation(self.bn6(x)).squeeze(2)

        # print(x.shape)

        stats = self.attention(x.permute(0, 2, 1).contiguous())

        feat = self.fc(stats)

        mu = self.fc_mu(feat)

        return feat, mu


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)

    MaxFeatureMap2D(max_dim=1)

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out
class Model(nn.Module):
    def __init__(self, args, nclasses=2):
        super(Model, self).__init__()
        # self.num_nodes = num_nodes
        self.enc_dim = args["enc_dim"]
        self.nclasses = nclasses
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.out = nn.Sequential(nn.Dropout(0.7),
                                 nn.Linear((args["feat_len"] // 16) * (60 // 16) * 32, 160),
                                 MaxFeatureMap2D(),
                                 nn.Linear(80, self.enc_dim))
        self.fc_mu = nn.Linear(self.enc_dim, nclasses) if nclasses >= 2 else nn.Linear(self.enc_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        feat = torch.flatten(x, 1)
        feat = self.out(feat)
        out = self.fc_mu(feat)

        return out



