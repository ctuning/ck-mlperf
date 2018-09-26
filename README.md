# Candidate Collective Knowledge benchmarks for MLPerf Inference

1. [Installation](#installation)
    1. [Install prerequisites](#installation-debian) (Debian-specific)
    1. [Install CK workflows](#installation-workflows) (universal)
1. [Benchmark MobileNets via TensorFlow Lite](#mobilenets-tflite)
1. [Benchmark MobileNets via TensorFlow (C++)](#mobilenets-tf-cpp)
1. [Benchmark MobileNets via TensorFlow (Python)](#mobilenets-tf-py)

<a name="installation"></a>
# Installation

<a name="installation-debian"></a>
## Debian (last tested with Ubuntu v18.04)

- Common tools and libraries.
- [Python](https://www.python.org/), [pip](https://pypi.org/project/pip/), [SciPy](https://www.scipy.org/), [Collective Knowledge](https://cknowledge.org) (CK).
- [[Optional]] [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/).

### Install common tools and libraries
```
$ sudo apt install gcc g++ git wget
$ sudo apt install libblas-dev liblapack-dev
```

### Install Python, pip, SciPy and CK
```
$ sudo apt install python3 python3-pip
$ sudo python3 -m pip install scipy
$ sudo python3 -m pip install ck
```
**NB:** CK also supports Python 2.

### [Optional] Install Android SDK and NDK

When using the TensorFlow Lite and TensorFlow (C++) benchmarks, you can optionally target Android API 23 (v6.0 "Marshmallow") devices using the `--target_os=android23-arm64` flag (or [similar](https://source.android.com/setup/start/build-numbers)).

On Debian Linux, you can install the [Android SDK](https://developer.android.com/studio/) and the [Android NDK](https://developer.android.com/ndk/) as follows:
```
$ sudo apt install android-sdk
$ sudo apt install google-android-ndk-installer
$ adb version
Android Debug Bridge version 1.0.36
Revision 1:7.0.0+r33-2
```

<a name="installation-workflows"></a>
## Install CK workflows

### Pull CK repositories
```
$ ck pull repo:ck-tensorflow
```

### Install a small dataset (500 images)
```
$ ck pull repo:ck-caffe --url=https://github.com/dividiti/ck-caffe
$ ck install package:imagenet-2012-val-min 
```
**NB:** ImageNet dataset descriptions are contained in [CK-Caffe](https://github.com/dividiti/ck-caffe) for historic reasons.

<a name="mobilenets-tflite"></a>
# MobileNets via TensorFlow Lite

This demo runs MobileNets ([v1](https://arxiv.org/abs/1704.04861) and [v2](https://arxiv.org/abs/1801.04381)) via [TensorFlow Lite](https://www.tensorflow.org/lite/).

*NB:** See [`program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite) for more details.


### Install TensorFlow Lite (TFLite)

Install TFLite from source:
```
$ ck install package:lib-tflite-0.1.7-src-static [--target_os=android23-arm64]
```

You can also install TFLite from a prebuilt binary package for your target e.g.:
```
$ ck list package:lib-tflite-prebuilt*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 [--target_os=android23-arm64]
```

### Install MobileNets models for TFLite

Select one of the 38 MobileNets models compatible with TFLite:
```
$ ck install package --tags=tensorflowmodel,mobilenet,tflite
```

### Compile the TFLite image classification client 
```
$ ck compile program:image-classification-tflite [--target_os=android23-arm64]
```

### Run the TFLite image classification client 

Run the client (if required, connect an Android device to your host machine via USB):
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
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
```

<a name="mobilenets-tf-cpp"></a>
# MobileNets via TensorFlow (C++)

**NB:** See [`program:image-classification-tf-cpp`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-cpp) for more details.

### Install TensorFlow (C++)

Install TensorFlow (C++) from source:
```
$ ck install package:lib-tensorflow-1.9.0-src-static
```

### Install MobileNets models for TensorFlow (C++)

Select one of the 38 MobileNets models compatible with TensorFlow (C++):
```
$ ck install package --tags=tensorflowmodel,mobilenet,frozen
```

### Compile the TensorFlow (C++) image classification client
```
$ ck compile program:image-classification-tf-cpp
```

### Run the TensorFlow (C++) image classification client
```
$ ck run program:image-classification-tf-cpp
...
*** Dependency 3 = weights (TensorFlow model and weights):
    ...
    Resolved. CK environment UID = b4fab4037b14a0b9 (version 2_1.4_224)
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.17 - (62) n01744401 rock python, rock snake, Python sebae
0.17 - (54) n01729322 hognose snake, puff adder, sand viper
0.10 - (58) n01737021 water snake
0.06 - (60) n01740131 night snake, Hypsiglena torquata
0.04 - (63) n01748264 Indian cobra, Naja naja
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.108859s
All images loaded in 0.005605s
All images classified in 0.481788s
Average classification time: 0.481788s
Accuracy top 1: 0.0 (0 of 1)
Accuracy top 5: 0.0 (0 of 1)
--------------------------------
```

<a name="mobilenets-tf-py"></a>
# MobileNets via TensorFlow (Python)

**NB:** See [`program:image-classification-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-py) for more details.

### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package (requires system `protobuf`):
```
$ sudo python3 -m pip install -U protobuf
$ ck install package:lib-tensorflow-1.10.1-cpu
```
or from source:
```
$ ck install package:lib-tensorflow-1.10.1-src-cpu
```

### Install MobileNets models for TensorFlow (Python)

Select one of the 54 MobileNets models compatible with TensorFlow (Python):
```
$ ck install package --tags=tensorflowmodel,mobilenet --no_tags=mobilenet-all
```

**NB:** This excludes "uber" packages which can be used to install all models in the sets `v1-2018-02-22` (16 models), `v1[-2018-06-14]` (16 models) and `v2` (22 models) in one go:
```
$ ck search package --tags=tensorflowmodel,mobilenet-all
ck-tensorflow:package:tensorflowmodel-mobilenet-v1-2018_02_22
ck-tensorflow:package:tensorflowmodel-mobilenet-v2
ck-tensorflow:package:tensorflowmodel-mobilenet-v1
```

### Run the TensorFlow (Python) image classification client
```
$ ck run program:image-classification-tf-py
...
*** Dependency 4 = weights (TensorFlow-Python model and weights):
    ...
    Resolved. CK environment UID = b4fab4037b14a0b9 (version 2_1.4_224)
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.38 - (65) n01751748 sea snake
0.19 - (54) n01729322 hognose snake, puff adder, sand viper
0.13 - (58) n01737021 water snake
0.12 - (62) n01744401 rock python, rock snake, Python sebae
0.03 - (60) n01740131 night snake, Hypsiglena torquata
---------------------------------------

Summary:
-------------------------------
Graph loaded in 1.933857s
All images loaded in 0.002172s
All images classified in 0.359537s
Average classification time: 0.359537s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```
