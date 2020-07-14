# MLPerf Inference v0.7 - Image Classification

## Table of Contents

1. [Prerequisites](#prereqs)
    1. [CK](#ck)
    1. [Inference engines](#inference_engines)
    1. [Preprocessed datasets](#preprocessed_datasets)
    1. [LoadGen config files](#loadgen_configs)
1. [Usage](#usage)

<a name="prereqs"></a>
# Prerequisites

<a name="ck"></a>
## Collective Knowledge

```bash
$ export CK_PYTHON=python3
$ ${CK_PYTHON} -m pip install ck
$ ck pull repo:ck-mlperf
$ ck pull repo --url=https://github.com/arm-software/armnn-mlperf
```

<a name="inference_engines"></a>
## Inference engines

The `run.sh` script assumes two inference engine: TFLite (`tflite`) and ArmNN (`armnn`).
Their latest versions are specified in the script:

```bash
$ grep inference_engine_version= * -B1 -n
run.sh-223-  if [ "${inference_engine}" == "tflite" ]; then
run.sh:224:    inference_engine_version="v2.1.1" # "v2.2.0
--
run.sh-227-  elif [ "${inference_engine}" == "armnn" ]; then
run.sh:228:    inference_engine_version="rel.20.05"
```

The inference engines should be installed using the above versions as follows.

### TFLite

```bash
$ ck install package --tags=lib,tflite,v2.1.1
```

### ArmNN

To parse TFLite models, ArmNN should be built with the TFLite frontend.

#### OpenCL backend
If your board has an Arm Mali GPU (e.g. Linaro HiKey960 or Firefly RK3399),
build ArmNN with the OpenCL backend:
```bash
$ ck install package --tags=lib,armnn,rel.20.05,tflite,neon,opencl
```

#### Neon backend
Otherwise, your board should support Arm Neon vector extensions,
so build ArmNN with the Neon backend only:
```bash
$ ck install package --tags=lib,armnn,rel.20.05,tflite,neon
```

<a name="preprocessed_datasets"></a>
## Preprocessed datasets

[Preprocess](https://github.com/ARM-software/armnn-mlperf#preprocess-on-an-x86-machine-and-detect-on-an-arm-dev-board) the ImageNet validation dataset on an x86 machine and copy to your Arm board.

### 224
```bash
$ ck detect soft:dataset.imagenet.preprocessed \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.224/ILSVRC2012_val_00000001.rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.224,universal
```

### 192
```bash
$ ck detect soft:dataset.imagenet.preprocessed \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.192/ILSVRC2012_val_00000001.rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.192,universal
```

### 160
```bash
$ ck detect soft:dataset.imagenet.preprocessed \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.160/ILSVRC2012_val_00000001.rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.160,universal
```

### 128
```bash
$ ck detect soft:dataset.imagenet.preprocessed \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.128/ILSVRC2012_val_00000001.rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.128,universal
```

### 96
```bash
$ ck detect soft:dataset.imagenet.preprocessed \
--full_path=/datasets/dataset-imagenet-preprocessed-using-opencv-crop.875-full-inter.linear-side.96/ILSVRC2012_val_00000001.rgb8 \
--extra_tags=using-opencv,crop.875,full,inter.linear,side.96,universal
```

<a name="loadgen_config"></a>
## LoadGen config files

### TFLite

```bash
$ ck detect soft --tags=config,loadgen,image-classification-tflite
```

### ArmNN

```bash
$ ck detect soft --tags=config,loadgen,image-classification-armnn-tflite
```

<a name="usage"></a>
# Usage

```bash
$ cd `ck find ck-mlperf:script:mlperf-inference-v0.7.image-classification`
```

| Parameter | Values | Default | Comment |
|-|-|-|-|
| `CK_DIVISION`| `closed`, `open` | `closed` | Workload selection. |
| `CK_MODE`| `performance`, `accuracy` | `performance` | Execution mode selection. |
| `CK_DATASET_SIZE`| positive integer | `50000` | Number of samples in the accuracy mode. |
| `CK_USE_LOADGEN` | `YES`, `NO` | `YES` | Use MLPerf LoadGen API. |
| `CK_DRY_RUN` | `YES`, `NO` | `NO` | Print commands but do not execute. |

<a name="performance"></a>
## Performance

```bash
$ CK_DIVISION=open CK_MODE=performance ./run.sh
```

<a name="accuracy"></a>
## Accuracy

### 50,000 images
```bash
$ CK_DIVISION=open CK_MODE=accuracy CK_DATASET_SIZE=50000 ./run.sh
```

### 500 images
```bash
$ CK_DIVISION=open CK_MODE=accuracy CK_DATASET_SIZE=500 ./run.sh
```
