import torch
import torch.nn as nn
from models.common import Conv, C3

__all__ = ['PAFPN']


class PAFPN(nn.Module):
    """
    width_mul: layer channel multiply
    depth_mul: model depth multiply
    feat_channels: base channel of input feature
    """
    def __init__(
            self,
            width_mul=1.0,
            depth_mul=1.0,
            feat_channels=(256, 512, 1024),
            act='silu'
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.base_depth = max(round(depth_mul * 3), 1)

        # top-down  p5 -> p4
        self.lateral_conv1 = Conv(int(feat_channels[2] * width_mul), int(feat_channels[1] * width_mul), 1, 1, act=act)
        self.C3_p4 = C3(int(2 * feat_channels[1] * width_mul),
                        int(feat_channels[1] * width_mul),
                        number=self.base_depth,
                        shortcut=False,
                        act=act)
        # top-down p4 -> p3
        self.reduce_conv0 = Conv(int(feat_channels[1] * width_mul), int(feat_channels[0] * width_mul), 1, 1, act=act)
        self.C3_p3 = C3(int(2 * feat_channels[0] * width_mul),
                        int(feat_channels[0] * width_mul),
                        number=self.base_depth,
                        shortcut=False,
                        act=act)

        # bottom-up N3 -> N4
        self.bu_conv0 = Conv(int(feat_channels[0] * width_mul), int(feat_channels[0] * width_mul), 3, 2, act=act)
        self.C3_n3 = C3(int(2 * feat_channels[0] * width_mul),
                        int(feat_channels[1] * width_mul),
                        number=self.base_depth,
                        shortcut=False,
                        act=act)

        # bottom-up N4 -> N5
        self.bu_conv1 = Conv(int(feat_channels[1] * width_mul), int(feat_channels[1] * width_mul), 3, 2, act=act)
        self.C3_n4 = C3(int(2 * feat_channels[1] * width_mul),
                        int(feat_channels[2] * width_mul),
                        number=self.base_depth,
                        shortcut=False,
                        act=act)

    def forward(self, features):
        [x0, x1, x2] = [features[f] for f in features]

        # top-down-fpn-p4
        fpn_out1 = self.lateral_conv1(x2)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x1], 1)
        f_out1 = self.C3_p4(f_out1)

        # top-down-fpn-p3
        fpn_out0 = self.reduce_conv0(f_out1)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x0], 1)
        pan_out0 = self.C3_p3(f_out0)

        # bottom-up-pan-n3
        p_out1 = self.bu_conv0(pan_out0)
        p_out1 = torch.cat([p_out1, fpn_out0], 1)
        pan_out1 = self.C3_n3(p_out1)

        # bottom-up-pfn-n4
        p_out2 = self.bu_conv1(pan_out1)
        p_out2 = torch.cat([p_out2, fpn_out1], 1)
        pan_out2 = self.C3_n4(p_out2)

        return [pan_out0, pan_out1, pan_out2]


if __name__ == "__main__":

    features = {
        'dark3': torch.randn(1, 128, 80, 80),
        'dark4': torch.randn(1, 256, 40, 40),
        'dark5': torch.randn(1, 512, 20, 20),
    }
    # yolov5-l
    # neck = PAFPN(width_mul=1.0,
    #              depth_mul=1.0,
    #              feat_channels=(256, 512, 1024),
    #              act='silu')
    neck = PAFPN(width_mul=0.5,
                 depth_mul=0.33,
                 feat_channels=(256, 512, 1024),
                 act='silu')

    out = neck(features)
    [print(f.shape) for f in out]


