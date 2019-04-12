# Collective Knowledge workflows for MLPerf

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![automation](https://github.com/ctuning/ck-guide-images/blob/master/ck-artifact-automated-and-reusable.svg)](http://cTuning.org/ae)

[![DOI](https://zenodo.org/badge/149591037.svg)](https://zenodo.org/badge/latestdoi/149591037)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Linux/MacOS: [![Travis Build Status](https://travis-ci.org/ctuning/ck-mlperf.svg?branch=master)](https://travis-ci.org/ctuning/ck-mlperf)

## MLPerf Inference v0.5

| Task | Model | Framework | CK workflow |
|-|-|-|-|
| Object Classification | MobileNet | [TFLite](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tflite/README.md) | yes |
|                       |           | [TF (C++)](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-cpp/README.md) | yes |
|                       |           | [TF (Python)](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-py/README.md) | yes |
|                       |           | [ONNX](https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/onnx/README.md) | yes |
| Object Classification | ResNet | TFLite | yes |
|                       |        | TF (C++) | yes |
|                       |        | TF (Python) | no |
|                       |        | ONNX | no |
| Object Detection | SSD-MobileNet | TFLite | yes |
|                  |               | TF (Python) | yes |
|                  |               | PyTorch | no |
| Object Detection | SSD-ResNet | PyTorch | no |
| Machine Translation | GNMT | TF (Python) | no |
|                     |      | PyTorch | no |

## MLPerf Training v0.6
**TODO**
