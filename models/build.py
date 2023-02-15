
import numpy as np
import torch
import torch.nn as nn


from models.network_blocks import Conv, DWConv
from models import YOLO


__all__ = ['DetectorModel']


class BaseModel(nn.Module):
    def __init__(self, profile=False, visualize=False):
        super().__init__()
        self.profile = profile
        self.visualze = visualize

    # YOLOv5 base model
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            pass
        return self._forward_once(x, profile=profile, visualize=visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):

        out = self.model(x)

        return out

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = self.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def fuse_conv_and_bn(self, conv, bn):
        # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              dilation=conv.dilation,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


class DetectorModel(BaseModel):
    def __init__(self, config, ckpt_path, profile=False, visualize=False, export=False):
        super().__init__(profile=profile, visualize=visualize)
        self.ckpt_path = ckpt_path
        self.model = YOLO(config, export=export)
        self.load_weights()

    def load_weights(self):
        ckpt = torch.load(self.ckpt_path)
        state_dict = ckpt['state_dict'] if ckpt.get('state_dict') else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        print(f'Successful load state dict from {self.ckpt_path}')

    def warmup(self, img_size=(1, 3, 640, 640)):
        """
        Warmup model by running inference once
        :param img_size:
        :return:
        """
        img = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(img)
