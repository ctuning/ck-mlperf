# Collective Knowledge workflows for MLPerf

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![automation](https://github.com/ctuning/ck-guide-images/blob/master/ck-artifact-automated-and-reusable.svg)](http://cTuning.org/ae)

[![DOI](https://zenodo.org/badge/149591037.svg)](https://zenodo.org/badge/latestdoi/149591037)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Linux/MacOS: [![Travis Build Status](https://travis-ci.org/ctuning/ck-mlperf.svg?branch=master)](https://travis-ci.org/ctuning/ck-mlperf)

## MLPerf Inference v0.5

| Task | Model | Owner | Contributors | Implementation | CK workflow |
|-|-|-|-|-|-|
| Object Classification | MobileNet     | dividiti  | dividiti  | [TFLite](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tflite) | yes |
|                       |               |           |           | [TF (C++)](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-cpp) | yes |
|                       |               |           |           | [TF (Python)](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-py) | yes |
|                       |               |           | NVIDIA, Microsoft | [ONNX](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/onnx) | [**coming soon!**](https://github.com/mlperf/inference/pull/76) |
| Object Classification | ResNet        | Microsoft | Microsoft | [ONNX](https://github.com/mlperf/inference/blob/master/cloud/image_classification) | no |
|                       |               |           |           | [TF (Python)](https://github.com/mlperf/inference/blob/master/cloud/image_classification) | no |
|                       |               |           | dividiti  | [TF (C++)](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-cpp#install-the-resnet-model) | yes |
|                       |               |           |           | [TFLite](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tflite#install-the-resnet-model) | yes |
|                       |               |           |           | [ONNX](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/onnx#install-the-resnet-model) | [**coming soon!**](https://github.com/mlperf/inference/pull/76) |
| Object Detection      | SSD-MobileNet | dividiti  | dividiti  | [TFLite](https://github.com/mlperf/inference/tree/master/edge/object_detection/ssd_mobilenet/tflite) | yes |
|                       |               |           |           | [TF (Python)](https://github.com/mlperf/inference/tree/master/edge/object_detection/ssd_mobilenet/tf-py) | yes |
|                       |               |           | Facebook  | [PyTorch](https://github.com/mlperf/inference/tree/master/edge/object_detection/ssd_mobilenet/pytorch) | no |
| Object Detection      | SSD-ResNet    | Habana    | NVIDIA    | [PyTorch](https://github.com/mlperf/inference/tree/master/cloud/single_stage_detector/pytorch) | no |
|                       |               |           | GM, dividiti | TF | **coming soon!** |
| Machine Translation   | GNMT          | Intel     | Intel     | [TF (Python)](https://github.com/mlperf/inference/blob/master/cloud/translation/gnmt/tensorflow) | no |
|                       |               |           |           | [PyTorch](https://github.com/mlperf/inference/blob/master/cloud/translation/gnmt/pytorch) | no |

## MLPerf Training v0.6
**TODO**
