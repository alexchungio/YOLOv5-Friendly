# YOLOv5-Friendly

Implement a friendly version of YOLOv5 by reconstruct [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repo.

## model convert
* install yolov5 repo [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* install yolov5 office model from [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* execute yolov5_fo_friendly.py under the ultralytics/yolov5 repo
```commandline
python tools/model_convert/yolov5_fo_friendly.py --src yolov5s.pt --dst yolov5s_friendly.pt
```
## inference
```commandline
python tools/detector.py
```
### show result(with yolov5s_friendly)

<div align="center">
    <img src=output/detect/exp/chair.jpg alt="chair" width="800"/>
    <img src=output/detect/exp/zidane.jpg alt="zidane" width="800"/>
</div>

## Log
### the official release code and the model not alignment
* adjust BN default param: `eps=0.001`, `momentum=0.03`

## TODO
* train
* eval

## References
[1] https://github.com/open-mmlab/mmyolo

[2] https://github.com/ultralytics/yolov5

[3] https://github.com/Megvii-BaseDetection/YOLOX

