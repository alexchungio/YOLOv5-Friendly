import os
import yaml
import torch
import torch.nn as nn
from models.backbones import *
from models.neck import *
from models.head import *

__all__ = ['YOLO']


class YOLO(nn.Module):
    def __init__(self, config:str, export=False):
        super().__init__()

        self.config = self.parse_yaml(config)
        self.backbone, self.neck, self.head = self.build_network(self.config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)

        return out

    def build_network(self, config, export=False):
        BACKBONE = eval(config.get('backbone', 'CSPDarknet'))
        NECK = eval(config.get('neck', 'PAFPN'))
        HEAD = eval(config.get('head', 'YOLOHead'))

        num_classes = config.get('num_classes', 80)
        anchors = config.get('anchors')
        width_mul = config.get('width_multiple', 1.0)
        depth_mul = config.get('width_multiple', 1.0)
        stride = config.get('stride', (8, 16, 32))
        out_features = config.get('out_features', ('dark3', 'dark4', 'dark5'))
        out_channels = config.get('out_channels', (256, 512, 1024))
        act = config.get('activate', 'silu')

        # build backbone
        backbone = BACKBONE(width_mul=width_mul,
                            depth_mul=depth_mul,
                            out_features=out_features,
                            act=act)
        # build neck
        neck = NECK(width_mul=width_mul,
                    depth_mul=depth_mul,
                    feat_channels=out_channels,
                    act=act)
        # build head
        head = HEAD(num_classes=num_classes,
                    width_mul=width_mul,
                    anchors=anchors,
                    stride=stride,
                    feature_channels=out_channels,
                    inplace=True,
                    export=export)

        return backbone, neck, head

    def parse_yaml(self, yaml_file):

        f = open(yaml_file, encoding='ascii')

        return yaml.safe_load(f)


if __name__ == "__main__":
    config = '/Users/alex/Documents/code/YOLOv5-Friendly/config/yolov5s_p5.yaml'
    model = YOLO(config)

    # training
    dummy_input = torch.randn(1, 3, 640, 640)
    model.train()
    out = model(dummy_input)
    [print(o.shape) for o in out]

    # eval
    model.eval()
    out = model(dummy_input)
    print(out[0].shape)