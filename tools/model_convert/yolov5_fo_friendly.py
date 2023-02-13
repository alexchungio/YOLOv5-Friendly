"""
acknowledge from https://github.com/open-mmlab/mmyolo/blob/main/tools/model_converters/yolov5_to_mmyolo.py
"""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict_p5 = {
    # backbone
    # stage-1
    'model.0': 'backbone.stem',
    # stage-2
    'model.1': 'backbone.dark2.0',
    'model.2': 'backbone.dark2.1',
    # stage-3
    'model.3': 'backbone.dark3.0',
    'model.4': 'backbone.dark3.1',
    # stage-4
    'model.5': 'backbone.dark4.0',
    'model.6': 'backbone.dark4.1',
    # stage-5
    'model.7': 'backbone.dark5.0',
    'model.8': 'backbone.dark5.1',
    # sppf
    'model.9.cv1': 'backbone.dark5.2.conv1',
    'model.9.cv2': 'backbone.dark5.2.conv2',
    # neck
    'model.10': 'neck.lateral_conv1',
    'model.13': 'neck.C3_p4',

    'model.14': 'neck.reduce_conv0',
    'model.17': 'neck.C3_p3',

    'model.18': 'neck.bu_conv0',
    'model.20': 'neck.C3_n3',

    'model.21': 'neck.bu_conv1',
    'model.23': 'neck.C3_n4',

    # head
    'model.24.m': 'head.blocks',
}

convert_dict_p6 = {}


def convert(src, dst):
    """Convert keys in pretrained YOLOv5 models to mmyolo style."""
    if src.endswith('6.pt'):
        convert_dict = convert_dict_p6
        is_p6_model = True
        print('Converting P6 model')
    else:
        convert_dict = convert_dict_p5
        is_p6_model = False
        print('Converting P5 model')
    try:
        yolov5_model = torch.load(src)['model']
        blobs = yolov5_model.state_dict()
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the ultralytics/yolov5 repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.')
    state_dict = OrderedDict()

    for key, weight in blobs.items():

        num, module = key.split('.')[1:3]
        if (is_p6_model and
            (num == '11' or num == '33')) or (not is_p6_model and
                                              (num == '9' or num == '24')):
            if module == 'anchors':
                continue
            prefix = f'model.{num}.{module}'
        else:
            prefix = f'model.{num}'

        new_key = key.replace(prefix, convert_dict[prefix])

        if '.m.' in new_key:
            new_key = new_key.replace('.m.', '.blocks.')
            new_key = new_key.replace('.cv', '.conv')
        else:
            new_key = new_key.replace('.cv1', '.conv1')
            new_key = new_key.replace('.cv2', '.conv2')
            new_key = new_key.replace('.cv3', '.conv3')

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


# Note: This script must be placed under the yolov5 repo to run.
def main():

    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolov5s.pt', help='src yolov5 model path')
    parser.add_argument('--dst', default='yolov5s_friendly.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()