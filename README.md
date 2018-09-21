# Candidate Collective Knowledge benchmarks for MLPerf Inference

This demo runs MobileNets ([v1](https://arxiv.org/abs/1704.04861) and [v2](https://arxiv.org/abs/1801.04381)) via [TensorFlow Lite](https://www.tensorflow.org/lite/) on Android devices.

## Install prerequisites on Ubuntu (last tested with 18.04)

- Python and [pip](https://pypi.org/project/pip/)
- [Collective Knowledge](https://cknowledge.org) (CK)
- [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/)

### Install common tools and libraries

```
$ sudo apt install git wget
$ sudo apt install python3 python3-pip
$ sudo apt install libblas-dev liblapack-dev
```

### Install CK with Python 3
```
$ sudo python3 -m pip install ck
```
**NB:** CK also supports Python 2.

### Install Android SDK and NDK
```
$ sudo apt install android-sdk
$ adb version
Android Debug Bridge version 1.0.36
Revision 1:7.0.0+r33-2
$ sudo apt install google-android-ndk-installer
```

## Install CK workflows

### Pull CK repositories
```
$ ck pull repo:ck-tensorflow
```

### Install a small dataset (500 images)
```
$ ck pull repo:ck-caffe --url=https://github.com/dividiti/ck-caffe
$ ck pull package:imagenet-2012-val-min 
```
**NB:** ImageNet dataset descriptions are contained in [CK-Caffe](https://github.com/dividiti/ck-caffe) for historic reasons.

### Install TensorFlow Lite

List available TensorFlow Lite packages:
```
$ ck list package:*tflite*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
lib-tflite-0.1.7-src-static
```

Install TFLite either from a prebuilt binary package:
```
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 --target_os=android23-arm64
```
or from source:
```
$ ck install package:lib-tflite-0.1.7-src-static --target_os=android23-arm64
```

### Install MobileNets models for TFLite

Select one of the 38 MobileNets models:
```
$ ck install package --tags=mobilenet,tflite
```

### Compile the TFLite image classification client 
```
$ ck compile program:image-classification-tflite --target_os=android23-arm64
```

### Run the TFLite image classification client 

Connect an Android device via USB and run the client:
```
$ ck run program:image-classification-tflite --target_os=android23-arm64
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.42 - (65) n01751748 sea snake
0.20 - (54) n01729322 hognose snake, puff adder, sand viper
0.14 - (58) n01737021 water snake
0.06 - (62) n01744401 rock python, rock snake, Python sebae
0.03 - (60) n01740131 night snake, Hypsiglena torquata
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.001319s
All images loaded in 0.007423s
All images classified in 0.202271s
Average classification time: 0.202271s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------


Execution time: 0.233 sec.
```
