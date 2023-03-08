import argparse
import os
import platform

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models import DetectorModel
from models.head import YOLOHead
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_load)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == 'Darwin'  # macOS environment


def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_onnx(model, img, ckpt_path, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements('onnx>=1.12.0')
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    onnx_path = ckpt_path.with_suffix('.onnx')

    input_names = ['image']
    output_names = ['output']
    if dynamic:
        dynamic = {'image': {0: 'batch', 2: 'height', 3: 'width'},
                   'output': {0: 'batch', 1: 'anchors'}}  # shape(1,3,640,640)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        img.cpu() if dynamic else img,
        onnx_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    meta_data = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in meta_data.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, onnx_path)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_path)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return onnx_path, model_onnx


@try_export
def export_coreml(model, img, ckpt_path, int8, half, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    check_requirements('coremltools')
    import coremltools as ct

    LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
    coreml_path = ckpt_path.with_suffix('.mlmodel')

    ts = torch.jit.trace(model, model, strict=False)  # TorchScript model
    ct_model = ct.convert(ts, inputs=(ct.ImageType('image', shape=img.shape, scale=1 / 255, bias=[0, 0, 0])))
    bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
    if bits < 32:
        if MACOS:  # quantization only supported on macOS
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress numpy==1.20 float warning
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        else:
            print(f'{prefix} quantization only supported on macOS, skipping...')
    ct_model.save(coreml_path)
    return coreml_path, ct_model


@try_export
def export_engine(model, img, ckpt_path, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert img.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, img, ckpt_path, 12, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, img, ckpt_path, 12, dynamic, simplify)  # opset 12
    onnx = ckpt_path.with_suffix('.onnx')

    LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    engine_path = ckpt_path.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if img.shape[0] <= 1:
            LOGGER.warning(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *img.shape[1:]), (max(1, img.shape[0] // 2), *img.shape[1:]), img.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {engine_path}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(engine_path, 'wb') as t:
        t.write(engine.serialize())
    return engine_path, None


@smart_inference_mode()
def run(
        data_cfg=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        model_cfg=ROOT / 'config/model/yolov5s_p5.yaml',
        weights=ROOT / 'weights/yolov5s.pt',  # weights path
        img_size=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        formats =('onnx'),  # output formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
):
    t = time.time()
    formats = [x.lower() for x in formats]  # to lowercase
    formats_lib = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in formats for x in formats_lib]
    assert sum(flags) == len(formats), f'ERROR: Invalid --include {formats}, valid --include arguments are {formats_lib}'
    onnx, engine, coreml = flags  # export booleans
    ckpt_path = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)

    # Load PyTorch model
    data_cfg = yaml_load(data_cfg)
    model_cfg = yaml_load(model_cfg)
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = DetectorModel(model_cfg, ckpt_path=ckpt_path, strict=True, export=True)
    model.fuse().eval()
    model.stride = model_cfg['stride']
    model.names = data_cfg['names']

    # Checks
    img_size *= 2 if len(img_size) == 1 else 1  # expand

    # Input
    img_size = [check_img_size(x, int(max(model.stride))) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, YOLOHead):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    # profile run
    y = model(img)

    if half and not coreml:
        im, model = img.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {'stride': int(max(model.stride,)), 'names': model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {ckpt_path} with output shape {shape} ({file_size(ckpt_path):.1f} MB)")

    # Exports
    filename = [''] * len(formats_lib)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if engine:  # TensorRT required before ONNX
        filename[0], _ = export_engine(model, img, ckpt_path, half, dynamic, simplify, workspace, verbose)
    if onnx:  # OpenVINO requires ONNX
        filename[1], _ = export_onnx(model, img, ckpt_path, opset, dynamic, simplify)
    if coreml:  # CoreML
        filename[2], _ = export_coreml(model, img, ckpt_path, int8, half)

    # Finish
    filename = [str(x) for x in filename if x]  # filter out '' and None
    if any(filename):
        cls, det, seg = False, True, False
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path('segment' if seg else 'classify' if cls else '')
        half = '--half' if half else ''  # --half FP16 inference arg
        LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                    f"\nResults saved to {colorstr('bold', ckpt_path.parent.resolve())}"
                    f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {filename[-1]} {half}"
                    f"\nValidate:        python {dir / 'val.py'} --weights {filename[-1]} {half}"
                    f'\nVisualize:       https://netron.app')
    return filename  # return list of exported files/dirs


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cfg', type=str, default=ROOT / 'config/model/yolov5s_p5.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--data-cfg', type=str, default=ROOT / 'config/dataset/coco128.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s_friendly.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['onnx'],
        help='onnx, engine, coreml')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
