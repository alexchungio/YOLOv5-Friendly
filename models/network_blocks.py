import cv2
import numpy as np
import pandas as pd
import json
import math
import warnings

import torch
import torch.nn as nn
from PIL import Image


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return act


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=None, group=1, dilation=1, act='silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, autopad(ksize, padding, dilation),
                              groups=group, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, dilation=1, act='selu'):
        super().__init__(in_channels, out_channels, ksize, stride, group=math.gcd(in_channels, out_channels),
                         dilation=dilation, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=0, padding_out=0):
        super().__init__(in_channels, out_channels, ksize, stride, padding, padding_out,
                         groups=math.gcd(in_channels, out_channels))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, channels, num_heads):
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc2 = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if in_channels != out_channels:
            self.conv = Conv(in_channels, out_channels)
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.transform = nn.Sequential(*(TransformerLayer(out_channels, num_heads) for _ in range(num_layers)))
        self.out_channels = out_channels

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.transform(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.out_channels, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, group=1, expansion=0.5, shortcut=True, act='silu'):
        super().__init__()
        c_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, c_, 1, 1, act=act)
        self.conv2 = Conv(c_, out_channels, 3, 1, group=group, act=act)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, group=1, expansion=0.5, numbers=1, shortcut=True):
        super().__init__()
        c_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, c_, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = Conv(2 * c_, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = get_activation('silu', inplace=True)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, group, expansion=1.0, shortcut=shortcut) for _ in range(numbers)))

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample decomposition factorization
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, group=1, expansion=1.0, shortcut=False, act='silu'):
        super().__init__()
        c_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, c_, (1, ksize), (1, stride), act=act)
        self.conv2 = Conv(c_, out_channels, (ksize, 1), (stride, 1), group=group, act=act)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, group=1, expansion=0.5, number=1, shortcut=True, act='silu'):
        super().__init__()
        c_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv(in_channels, c_, 1, 1, act=act)
        self.conv2 = Conv(in_channels, c_, 1, 1, act=act)
        self.conv3 = Conv(2 * c_, out_channels, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, group, expansion=1.0, shortcut=shortcut, act=act) for _ in range(number)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, in_channels, out_channels, group=1, expansion=0.5, number=1, shortcut=True, act='silu'):
        super().__init__(in_channels, out_channels, group, expansion, number, shortcut)
        c_ = int(out_channels * expansion)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, group, 1.0, shortcut=shortcut, act=act) for _ in range(number)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, in_channels, out_channels, group=1, expansion=0.5, number=1, shortcut=True):
        super().__init__(in_channels, out_channels, group, expansion, number, shortcut)
        c_ = int(out_channels * expansion)
        self.m = TransformerBlock(c_, c_, num_heads=4, num_layers=number)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, in_channels, out_channels, ksize=(5, 9, 13), group=1, expansion=0.5, number=1, shortcut=True,
                 act='silu'):
        super().__init__(in_channels, out_channels, group, expansion, number, shortcut)
        c_ = int(out_channels * expansion)
        self.m = SPP(c_, c_, ksize)


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, in_channels, out_channels, ksize=(5, 9, 13), act='silu'):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.conv1 = Conv(in_channels, c_, 1, 1, act=act)
        self.con2 = Conv(c_ * (len(ksize) + 1), out_channels, 1, 1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in ksize])

    def forward(self, x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_channels, out_channels, ksize=5, act='silu'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.conv1 = Conv(in_channels, c_, 1, 1, act=act)
        self.conv2 = Conv(c_ * 4, out_channels, 1, 1, act=act)
        self.max_pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)

    def forward(self, x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.max_pool(x)
            y2 = self.max_pool(y1)
            return self.conv2(torch.cat((x, y1, y2, self.max_pool(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=None, group=1, act='silu'):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, ksize, stride, padding, group, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)