# yolo-face-with-landmark

### 实现的功能
- 添加关键点检测分支，使用wing loss

## Installation
##### Clone and install
1. git clone https://github.com/ouyanghuiyu/yolo-face-with-landmark
2. 使用src/retinaface2yololandmark.py脚本将retinaface的标记文件转为yolo的格式使用,
3. 使用src/create_train.py 创建训练样本

## 训练
```
python train.py --net mbv3_large_75 --backbone_weights \
./pretrained/mobilenetv3-large-0.75-9632d2a8.pth --batch-size 16 
```

## 测试 
```
python evaluation_on_widerface.py
cd widerface_evaluate
python evaluation.py
```

## demo
```
python demo.py
```

## 精度
### Widerface测试
- 在wider face val精度（单尺度输入分辨率：**320*240**）
 
 方法|Easy|Medium|Hard|Flops
------|--------|----------|--------|--------
Retinaface-Mobilenet-0.25(Mxnet)  |0.745|0.553|0.232
mbv3large_1.0_yolov3(our)  |**0.861**|**0.781**|**0.387**|405M
mbv3large_1.0_yolov3_light(our)  |0.856|0.770|0.370|311M
mbv3large_0.75_yolov3(our)  |0.853|0.778|0.382|334M
mbv3large_0.75_yolov3_light(our)  |0.845|0.766|0.365|240M
mbv3samll_1.0_yolov3(our)  |0.798|0.696|0.3|185M
mbv3small_1.0_yolov3_light(our)  |0.759|0.662|0.300|91M
mbv3samll_0.75_yolov3(our)  |0.768|0.673|0.305|174M
mbv3small_0.75_yolov3_light(our)  |0.754|0.647|0.291|80M
- 在wider face val精度（单尺度输入分辨率：**640*480**） 

方法|Easy|Medium|Hard 
------|--------|----------|--------
Retinaface-Mobilenet-0.25(mxnet)  |0.879|0.807|0.481
mbv3large_1.0_yolov3(our)  |**0.900**|**0.882**|**0.707**
mbv3large_1.0_yolov3_light(our)  |0.900|0.874|0.683|
mbv3large_0.75_yolov3(our)  |0.886|0.871|0.694|
mbv3large_0.75_yolov3_light(our)  |0.881|0.862|0.678
mbv3samll_1.0_yolov3(our)  |0.856|0.827|0.602
mbv3small_1.0_yolov3_light(our)  |0.847|0.807|0.578
mbv3samll_0.75_yolov3(our)  |0.841|0.815|0.584
mbv3small_0.75_yolov3_light(our)  |0.832|0.796|0.553|
ps: 测试的时候,长边为320 或者 640 ,图像等比例缩放

## 测试
<p align="center"><img src="test_imgs/selfie.jpg"\></p>




## References
- [yolov3](https://github.com/ultralytics/yolov3)
