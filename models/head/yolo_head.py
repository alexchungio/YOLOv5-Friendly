import math
import torch
import torch.nn as nn
from models.network_blocks import Conv

__all__ = ['YOLOHead']


class YOLOHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            width_mul=1.0,
            anchors=None,
            stride=(8, 16, 32),
            feature_channels=(256, 512, 1024),
            act=None,
            inplace=True,
            export=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(anchors) # number of detection scale layer
        self.num_anchor = len(anchors[0]) // 2  # number of anchors
        self.output_channels = (num_classes + 5) * self.num_anchor # (80(classify) + 4(box) + 1(obj_conf)) * 3(anchor)
        self.stride = stride
        self.grid = [torch.empty(0) for _ in range(self.num_layers)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_layers)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_layers, -1, 2))
        # output head
        self.m = nn.ModuleList(Conv(int(feature_channels[i] * width_mul), self.output_channels, 1, act=act)
                               for i in range(self.num_layers))
        self.inplace = inplace
        self.export = export
        
    def forward(self, x):
        out = []
        # infer differ scale layer
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])
            # x(bs, 255, -, -) to x(bs, 3, -, -, 85)
            batch_size, _, num_y, num_x = x[i].shape
            x[i] = x[i].view(batch_size,
                             self.num_anchor,
                             self.num_classes + 5,
                             num_y,
                             num_x).permute(0, 1, 3, 4, 2).contiguous()

            # box encoder only inference
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(num_x, num_y, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                xy_wh_conf = torch.cat((xy, wh, conf), 4)
                out.append(xy_wh_conf.view(batch_size, self.num_anchor * num_x * num_y, self.num_classes + 5))

        return tuple(x) if self.training else (torch.cat(out, 1),) if self.export else (torch.cat(out, 1), x)

    def _make_grid(self, num_x, num_y, i=0):
        device = self.anchors[i].device
        type = self.anchors[i].dtype
        shape = 1, self.num_anchor, num_y, num_x, 2  # grid shape
        y, x = torch.arange(num_y, device=device, dtype=type), torch.arange(num_x, device=device, dtype=type)
        y_grid, x_grid = torch.meshgrid(y, x)
        grid = torch.stack((y_grid, x_grid), 2).expand(shape) - 0.5  # add grid offset, y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.num_anchor, 1, 1, 2)).expand(shape)

        return grid, anchor_grid


if __name__ == "__main__":
    anchors=[
        [10,13, 16,30, 33,23],
        [30,61, 62,45, 59,119],
        [116,90, 156,198, 373,326]
    ]

    # yolov5-s
    head = YOLOHead(anchors=anchors, width_mul=0.5)

    features = [
        torch.randn(1, 128, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 512, 20, 20),
    ]

    out = head(features)
    [print(f.shape) for f in out]




