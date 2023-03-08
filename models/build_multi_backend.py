
from collections import OrderedDict, namedtuple
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import torch
import torch.nn as nn
from PIL import Image

from.build import DetectorModel
from utils.general import (LOGGER, ROOT, check_requirements, check_version, xywh2xyxy, yaml_load, load_weight)


__all__ = ['DetectMultiBackend']


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', model_cfg=None, data_cfg=None, device=torch.device('cpu'),
                 dnn=False, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine

        super().__init__()
        weights = str(weights[0] if isinstance(weights, list) else weights)
        pt, onnx, engine, coreml = self._model_type(weights)
        fp16 &= pt or onnx or engine  # FP16
        nhwc = coreml  # channel format
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if pt:  # PyTorch
            model_cfg = yaml_load(model_cfg)
            data_cfg = yaml_load(data_cfg)
            model = DetectorModel(model_cfg)
            load_weight(model, weights, strict=True)
            model.num_classes = int(model_cfg['num_classes'])
            model.stride = max(max(model_cfg['stride']), stride)
            model.names = data_cfg['names']
            model.fuse().eval() if fuse else model.eval()
            model.half() if fp16 else model.float()
            stride = model.stride # model stride
            names = model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {weights} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(weights)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {weights} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(weights, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {weights} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {weights} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(weights)
        else:
            raise NotImplementedError(f'ERROR: {weights} is not a supported format')
        # class names
        if 'names' not in locals():
            names = yaml_load(data_cfg)['names'] if data_cfg else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            y = None

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, img_size=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.onnx, self.engine
        if any(warmup_types) and (self.device.type != 'cpu'):
            im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from tools.export import export_formats
        type_suffix = list(export_formats().Suffix)  # export suffixes
        types = [s in Path(p).name for s in type_suffix]
        return types

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None
