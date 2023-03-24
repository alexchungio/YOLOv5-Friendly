# YOLOv5-Friendly

Implement a friendly version of YOLOv5 by reconstruct [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repo.

## Train
* training(with single gpu)
```commandline
python tools/train.py
```
* ddp training(with multi gpu)
```commandline
bash tool/dist_train.sh
```

## Evaluation
```commandline
python tools/eval.py
```

## Inference
```commandline
python tools/detector.py
```
### show result(with yolov5s_friendly)

<div align="center">
    <img src=runs/detect/exp/sand.jpg alt="sand" width="800"/>
    <img src=runs/detect/exp/zidane.jpg alt="zidane" width="800"/>
</div>

## Export
```commandline
python tools/export.py --inplace --simplify
```

## Model and Result

| Model            | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | params<br><sup>(M) | FLOPs<br><sup>@640 (B) | Config                                 | Checkpoint                                                                                                   |
|------------------|-----------------------|----------------------|-------------------|--------------------|------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| YOLOv5n_friendly | 640                   | 28.0                 | 45.7              | **1.9**            | **4.5**                | [config](config/model/yolov5n_p5.yaml) | [download](https://github.com/alexchungio/YOLOv5-Friendly/releases/download/v1.0.0/yolov5n_friendly.pt) |
| YOLOv5s_friendly | 640                   | 37.4                 | 56.8              | 7.2                | 16.5                   | [config](config/model/yolov5s_p5.yaml) | [download](https://github.com/alexchungio/YOLOv5-Friendly/releases/download/v1.0.0/yolov5s_friendly.pt) |
| YOLOv5m_friendly | 640                   | 45.4                 | 64.1              | 21.2               | 49.0                   | [config](config/model/yolov5m_p5.yaml) | [download](https://github.com/alexchungio/YOLOv5-Friendly/releases/download/v1.0.0/yolov5m_friendly.pt) |
| YOLOv5l_friendly | 640                   | 49.0                 | 67.3              | 46.5               | 109.1                  | [config](config/model/yolov5l_p5.yaml) | [download](https://github.com/alexchungio/YOLOv5-Friendly/releases/download/v1.0.0/yolov5l_friendly.pt) |
| YOLOv5x_friendly | 640                   | 50.7                 | 68.9              | 86.7               | 205.7                  | [config](config/model/yolov5x_p5.yaml) | [download](https://github.com/alexchungio/YOLOv5-Friendly/releases/download/v1.0.0/yolov5x_friendly.pt) |


## Model Convert
* install yolov5 repo [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* install yolov5 office model from [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* execute yolov5_to_friendly.py under the ultralytics/yolov5 repo
```commandline
python tools/model_convert/yolov5_to_friendly.py --src yolov5s.pt --dst yolov5s_friendly.pt
```

## Log
### the BN value do not use default value
* adjust BN default param: `eps=0.001`, `momentum=0.03`
### training do not use ImageNet pretrained model
*all of our models are trained from scratch on COCO. We do not have any ImageNet trained models @glenn-jocher.*[reference](https://github.com/ultralytics/yolov5/issues/5422)

## TODO
- [x] Train
- [x] Evaluation
- [x] Export

## References
[1] https://github.com/open-mmlab/mmyolo

[2] https://github.com/ultralytics/yolov5

[3] https://github.com/Megvii-BaseDetection/YOLOX

