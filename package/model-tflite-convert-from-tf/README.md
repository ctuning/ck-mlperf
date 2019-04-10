# Install TFLite models by converting from TF models

A package that installs TFLite models by converting from TF models.

Currently, only the following [MLPerf](http://github.com/mlperf/inference)
Image Classification models are supported:
- [MobileNet non-quantized](#mobilenet)
- [MobileNet quantized](#mobilenet_quant).
- [ResNet](#resnet).

<a name="mobilenet"></a>
## MobileNet non-quantized
```
$ ck install package --tags=tflite,model,converted,mobilenet
```

<a name="mobilenet_quant"></a>
## MobileNet quantized
```
$ ck install package --tags=tflite,model,converted,mobilenet-quantized
```

<a name="resnet"></a>
## ResNet
```
$ ck install package --tags=tflite,model,converted,resnet
```
