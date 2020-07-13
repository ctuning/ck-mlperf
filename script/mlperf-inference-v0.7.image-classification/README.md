# MLPerf Inference v0.7 - Image Classification

## Table of Contents

1. [Prerequisites](#prereqs)
  1. [CK](#ck)
  1. [Inference engines](#inference_engines)
  1. [Preprocessed datasets](#preprocessed_datasets)
  1. [LoadGen config files](#loadgen_configs)

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
Otherwise, you board should support Arm Neon vector extensions,
so build ArmNN the Neon backend only:
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
