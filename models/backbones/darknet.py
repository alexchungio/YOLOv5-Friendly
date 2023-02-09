
import torch
import torch.nn as nn

from models.common import Conv, C3, SPPF

__all__ = ['CSPDarknet']


class DarkNet(nn.Module):
    pass


class CSPDarknet(nn.Module):
    """
    width_mul: layer channel multiple
    depth_mul:
    """
    def __init__(
        self,
        width_mul=1.0,  # 0.5
        depth_mul=1.0,  # 0.33
        out_features=("dark3", "dark4", "dark5"),
        act="silu",
    ):

        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(width_mul * 64)  # 64
        base_depth = max(round(depth_mul * 3), 1)  # 3

        self.stem = Conv(in_channels=3, out_channels=base_channels, ksize=6, padding=2, stride=2, act=act)

        # p2
        self.dark2 = nn.Sequential(
            Conv(in_channels=base_channels, out_channels=base_channels * 2, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 2,
               out_channels=base_channels * 2,
               number=base_depth)
        )

        # p3
        self.dark3 = nn.Sequential(
            Conv(in_channels=base_channels * 2, out_channels=base_channels * 4, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 4,
               out_channels=base_channels * 4,
               number=base_depth * 2)
        )

        # p4
        self.dark4 = nn.Sequential(
            Conv(in_channels=base_channels * 4, out_channels=base_channels * 8, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 8,
               out_channels=base_channels * 8,
               number=base_depth * 3)
        )

        # p5
        self.dark5 = nn.Sequential(
            Conv(in_channels=base_channels * 8, out_channels=base_channels * 16, ksize=3, padding=1, stride=2, act=act),
            C3(in_channels=base_channels * 16,
               out_channels=base_channels * 16,
               number=base_depth * 1),
            SPPF(in_channels=base_channels * 16, out_channels=base_channels * 16, ksize=5)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        out = self.dark5(x)
        outputs["dark5"] = out
        # return {k: v for k, v in outputs.items() if k in self.out_features}

        return out


if  __name__ == "__main__":

    dummy_input = torch.randn(1, 3, 640, 640)

    # yolov5-s
    model = CSPDarknet(width_mul=0.5, depth_mul=0.33)

    out = model(dummy_input)
    print(out.shape)






