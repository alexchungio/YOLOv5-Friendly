
import torch
import torch.nn as nn

from models.common import Conv, C3, SPPF

__all__ = ['CSPDarknet']


class DarkNet(nn.Module):
    pass


class CSPDarknet(nn.Module):
    """
    width_mul: layer channel multiple
    depth_mul: model depth multiple
    """
    def __init__(
        self,
        width_mul=1.0,
        depth_mul=1.0,
        out_features=("dark3", "dark4", "dark5"),
        act="silu",
    ):

        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(width_mul * 64)  # 64
        base_depth = max(round(depth_mul * 3), 1)  # 3

        # P1
        self.stem = Conv(in_channels=3, out_channels=base_channels, ksize=6, padding=2, stride=2, act=act)

        # p2
        self.dark2 = nn.Sequential(
            Conv(in_channels=base_channels, out_channels=base_channels * 2, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 2,
               out_channels=base_channels * 2,
               number=base_depth,
               act=act)
        )

        # p3
        self.dark3 = nn.Sequential(
            Conv(in_channels=base_channels * 2, out_channels=base_channels * 4, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 4,
               out_channels=base_channels * 4,
               number=base_depth * 2,
               act=act)
        )

        # p4
        self.dark4 = nn.Sequential(
            Conv(in_channels=base_channels * 4, out_channels=base_channels * 8, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 8,
               out_channels=base_channels * 8,
               number=base_depth * 3,
               act=act)
        )

        # p5
        self.dark5 = nn.Sequential(
            Conv(in_channels=base_channels * 8, out_channels=base_channels * 16, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 16,
               out_channels=base_channels * 16,
               number=base_depth * 1,
               act=act),
            SPPF(in_channels=base_channels * 16, out_channels=base_channels * 16, ksize=5, act=act)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)  # 640->320/2
        outputs["stem"] = x
        x = self.dark2(x)  # 320->160/4
        outputs["dark2"] = x
        x = self.dark3(x)  # 160->80/8
        outputs["dark3"] = x
        x = self.dark4(x)  # 80-> 0/16
        outputs["dark4"] = x
        out = self.dark5(x)  # 40->20/32
        outputs["dark5"] = out

        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == "__main__":

    dummy_input = torch.randn(1, 3, 640, 640)

    # yolov5-l
    # model = CSPDarknet(width_mul=1.0, depth_mul=1.0)
    # yolov5-s
    model = CSPDarknet(width_mul=0.5, depth_mul=0.33)
    out = model(dummy_input)

    [print(f'{name}: {m.shape}') for name, m in out.items()]







