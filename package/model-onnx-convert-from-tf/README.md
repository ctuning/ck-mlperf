# Install ONNX models by converting from TF models

A package that installs ONNX models by converting from TF models.

Currently, only the following [MLPerf](http://github.com/mlperf/inference)
Image Classification models are supported:
- [MobileNet](#mobilenet) (non-quantized).
- [ResNet](#resnet).

<a name="mobilenet"></a>
## MobileNet
### NHWC
```
$ ck install package --tags=onnx,model,mobilenet,converted,nhwc
```
### NCHW
```
$ ck install package --tags=onnx,model,mobilenet,converted,nchw
```

<a name="resnet"></a>
## ResNet
### NHWC
```
$ ck install package --tags=onnx,model,resnet,converted,nhwc
```
### NCHW
```
$ ck install package --tags=onnx,model,resnet,converted,nchw
```
