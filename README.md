# Collective Knowledge workflows for MLPerf

**All CK components for AI and ML are now collected in [one repository](https://github.com/ctuning/ai)!**

*This project is hosted by the [cTuning foundation (non-profit R&D organization)](https://cTuning.org).*

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![automation](https://github.com/ctuning/ck-guide-images/blob/master/ck-artifact-automated-and-reusable.svg)](http://cTuning.org/ae)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Linux/MacOS: [![Travis Build Status](https://travis-ci.org/ctuning/ck-mlperf.svg?branch=master)](https://travis-ci.org/ctuning/ck-mlperf)


## Table of Contents
- [Installation](#installation)
- [Inference v0.5](#inference_0_5) 
    - [Unofficial CK workflows](#unofficial)
    - [CK workflows for official application with Docker](#official_docker)
        - [Datasets](#datasets)
        - [Models](#models)
            - [ResNet](#resnet)
            - [MobileNet non-quantized](#mobilenet)
            - [MobileNet quantized](#mobilenet_quant)
            - [SSD-MobileNet non-quantized](#ssd_mobilenet)
            - [SSD-MobileNet quantized](#ssd_mobilenet_quant)
            - [SSD-ResNet](#ssd_resnet)
    - [CK workflows for official application without Docker](#official_native)
        - [Prerequisites](#prereqs)
        - [Modify](#modify_run_local) `run_local.sh`
        - [Use](#use_run_local) `run_local_sh`

- [Training v0.7](#training_0_7)


<a name="installation"></a>
## Installation

### Install CK
```bash
$ python -m pip install ck --user
$ ck version
V1.11.1
```

### Pull CK repositories
Pull repos (recursively, pulls `ck-env`, `ck-tensorflow`, etc.):
```bash
$ ck pull repo:ck-mlperf
```

<a name="inference_0_5"></a>
## MLPerf Inference v0.5

Using CK is optional for MLPerf Inference v0.5.

<a name="unofficial"></a>
### Unofficial CK workflows

We (unofficially) support two tasks out of three (i.e. except for Machine Translation).
Full instructions are provided in the official MLPerf Inference repository:
- [Image Classification](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection/optional_harness_ck/classification)
- [Object Detection](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection/optional_harness_ck/detection)

<a name="official_docker"></a>
### CK workflows for official application with Docker

You can run the official vision application with CK model and dataset packages.


<a name="datasets"></a>
#### Install datasets 

<a name="imagenet"></a>
##### ImageNet 2012 validation dataset
Download the original dataset and auxiliaries:
```bash
$ ck install package --tags=image-classification,dataset,imagenet,val,original,full
$ ck install package --tags=image-classification,dataset,imagenet,aux
```
Copy the labels next to the images:
```bash
$ ck locate env --tags=image-classification,dataset,imagenet,val,original,full
/home/dvdt/CK-TOOLS/dataset-imagenet-ilsvrc2012-val
$ ck locate env --tags=image-classification,dataset,imagenet,aux
/home/dvdt/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux
$ cp `ck locate env --tags=aux`/val.txt `ck locate env --tags=val`/val_map.txt
```

<a name="coco"></a>
##### COCO 2017 validation dataset
```bash
$ ck install package --tags=object-detection,dataset,coco,2017,val,original
$ ck locate env --tags=object-detection,dataset,coco,2017,val,original
/home/dvdt/CK-TOOLS/dataset-coco-2017-val
```

<a name="models"></a>
#### Install and run TensorFlow models

**NB:** It is currently necessary to create symbolic links if a model's file
name is different from the one hardcoded in the application for each profile.
For example, for the `tf-mobilenet` profile (which can be used both for the
non-quantized and quantized MobileNet TF models), the application specifies
`mobilenet_v1_1.0_224_frozen.pb` , but the quantized model's file is
`mobilenet_v1_1.0_224_quant_frozen.pb`.

<a name="resnet"></a>
##### ResNet
```bash
$ ck install package --tags=mlperf,image-classification,model,tf,resnet
$ export MODEL_DIR=`ck locate env --tags=model,tf,resnet`
$ export DATA_DIR=`ck locate env --tags=dataset,imagenet,val`
$ export EXTRA_OPS="--accuracy --count 50000 --scenario SingleStream"
$ ./run_and_time.sh tf resnet cpu
...
TestScenario.SingleStream qps=1089.79, mean=0.0455, time=45.880, acc=76.456, queries=50000, tiles=50.0:0.0447,80.0:0.0465,90.0:0.0481,95.0:0.0501,99.0:0.0564,99.9:0.0849
```

<a name="mobilenet"></a>
##### MobileNet non-quantized
```bash
$ ck install package --tags=mlperf,image-classification,model,tf,mobilenet,non-quantized
$ export MODEL_DIR=`ck locate env --tags=model,tf,mobilenet,non-quantized`
$ export DATA_DIR=`ck locate env --tags=dataset,imagenet,val`
$ export EXTRA_OPS="--accuracy --count 50000 --scenario Offline"
$ ./run_and_time.sh tf mobilenet cpu
...
TestScenario.Offline qps=352.92, mean=3.2609, time=4.534, acc=71.676, queries=1600, tiles=50.0:2.9725,80.0:4.0271,90.0:4.0907,95.0:4.3719,99.0:4.4811,99.9:4.5173
```

<a name="mobilenet_quant"></a>
##### MobileNet quantized
```bash
$ ck install package --tags=mlperf,image-classification,model,tf,mobilenet,quantized
$ ln -s `ck locate env --tags=mobilenet,quantized`/mobilenet_v1_1.0_224{_quant,}_frozen.pb`
$ export MODEL_DIR=`ck locate env --tags=model,tf,mobilenet,quantized`
$ export DATA_DIR=`ck locate env --tags=dataset,imagenet,val`
$ export EXTRA_OPS="--accuracy --count 50000 --scenario Offline"
$ ./run_and_time.sh tf mobilenet cpu
...
TestScenario.Offline qps=128.83, mean=7.5497, time=12.419, acc=70.676, queries=1600, tiles=50.0:7.8294,80.0:11.1442,90.0:11.7616,95.0:12.1174,99.0:12.9126,99.9:13.1641
```

<a name="ssd_mobilenet"></a>
##### SSD-MobileNet non-quantized
```bash
$ ck install package --tags=mlperf,object-detection,model,tf,ssd-mobilenet,non-quantized
$ ln -s `ck locate env --tags=model,tf,ssd-mobilenet,non-quantized`/{frozen_inference_graph.pb,ssd_mobilenet_v1_coco_2018_01_28.pb}
$ export MODEL_DIR=`ck locate env --tags=model,tf,ssd-mobilenet,non-quantized`
$ export DATA_DIR=`ck locate env --tags=dataset,coco,2017,val`
$ export EXTRA_OPS="--accuracy --count 5000 --scenario Offline"
$ ./run_and_time.sh tf ssd-mobilenet cpu
...
TestScenario.Offline qps=5.82, mean=8.0406, time=27.497, acc=93.312, mAP=0.235, queries=160, tiles=50.0:6.7605,80.0:10.3870,90.0:10.4632,95.0:10.4788,99.0:10.4936,99.9:10.5068
```

<a name="ssd_mobilenet_quant"></a>
##### SSD-MobileNet quantized
```bash
$ ck install package --tags=mlperf,object-detection,model,tf,ssd-mobilenet,quantized
$ ln -s `ck locate env --tags=model,tf,ssd-mobilenet,quantized`/{graph.pb,ssd_mobilenet_v1_coco_2018_01_28.pb}
$ export MODEL_DIR=`ck locate env --tags=model,tf,ssd-mobilenet,quantized`
$ export DATA_DIR=`ck locate env --tags=dataset,coco,2017,val`
$ export EXTRA_OPS="--accuracy --count 5000 --scenario Offline"
$ ./run_and_time.sh tf ssd-mobilenet cpu
...
TestScenario.Offline qps=5.46, mean=9.4975, time=29.310, acc=94.037, mAP=0.239, queries=160, tiles=50.0:7.9843,80.0:12.2297,90.0:12.3646,95.0:12.3965,99.0:12.4229,99.9:12.4351
```

<a name="ssd_resnet"></a>
##### SSD-ResNet
**TODO**


<a name="official_native"></a>
### CK workflows for official application without Docker

<a name="prereqs"></a>
#### Install prerequisites

To run the official vision app natively (i.e. without Docker), first install Python prerequisites such as OpenCV, TensorFlow and COCO Python API:
```bash
$ ck detect soft --tags=compiler,python --full_path=`which python3`
$ ck install package --tags=lib,tensorflow,v1.14,vcpu,vprebuilt
$ ck install package --tags=lib,python-package,cv2
$ ck install package --tags=tool,coco,api
```

Then, install the latest LoadGen package:
```bash
$ ck install package --tags=mlperf,inference,source,upstream.master
$ ck install package --tags=lib,python-package,absl
$ ck install package --tags=lib,python-package,mlperf,loadgen
```

**NB:** The most important thing during installation is to select the same version of Python 3 (if you have more than one registered with CK).
Check that each package "needs" exactly the same version of Python 3 after installation:
```bash
$ ck show env --tags=lib,tensorflow,v1.14,vcpu,vprebuilt
Env UID:         Target OS: Bits: Name:                              Version: Tags:
087035468886d589   linux-64    64 TensorFlow library (prebuilt, cpu) 1.14.0   64bits,channel-stable,host-os-linux-64,lib,needs-python,needs-python-3.6.7,target-os-linux-64,tensorflow,tensorflow-cpu,tf,tf-cpu,v1,v1.14,v1.14.0,vcpu,vprebuilt

$ ck show env --tags=lib,python-package,cv2
Env UID:         Target OS: Bits: Name:                 Version: Tags:
5f31d16b444d6b8c   linux-64    64 Python OpenCV library 3.6.7    64bits,cv2,host-os-linux-64,lib,needs-python,needs-python-3.6.7,opencv,python-package,target-os-linux-64,v3,v3.6,v3.6.7

$ ck show env --tags=tool,coco,api
Env UID:         Target OS: Bits: Name:            Version: Tags:
885a8f71bf1219da   linux-64    64 COCO dataset API master   64bits,api,coco,compiled-by-gcc,compiled-by-gcc-8.3.0,host-os-linux-64,needs-python,needs-python-3.6.7,target-os-linux-64,tool,v0,vmaster,vtrunk

$ ck show env --tags=lib,python-package,mlperf,loadgen
Env UID:         Target OS: Bits: Name:                            Version: Tags:
462592cb2beeaf63   linux-64    64 MLPerf Inference LoadGen library master   64bits,host-os-linux-64,lib,loadgen,mlperf,mlperf-loadgen,mlperf_loadgen,needs-python,needs-python-3.6.7,python-package,target-os-linux-64,v0,vmaster
```

<a name="modify_run_local"></a>
#### Modify `run_local.sh`

Modify the `run_local.sh` script under `v0.5/classification_and_detection` as follows:

```bash
$ git diff
diff --git a/v0.5/classification_and_detection/run_local.sh b/v0.5/classification_and_detection/run_local.sh
index 1262991..7597403 100755
--- a/v0.5/classification_and_detection/run_local.sh
+++ b/v0.5/classification_and_detection/run_local.sh
@@ -9,5 +9,5 @@ if [ ! -d $OUTPUT_DIR ]; then
     mkdir -p $OUTPUT_DIR
 fi
 
-python python/main.py --profile $profile $common_opt --model $model_path $dataset \
-    --output $OUTPUT_DIR $EXTRA_OPS $@
+ck virtual env --tag_groups="lib,tensorflow-cpu,v1.14,vcpu,vprebuilt lib,python-package,cv2 tool,coco lib,python-package,mlperf,loadgen" \
+--shell_cmd="python3.6 python/main.py --profile $profile $common_opt --model $model_path $dataset --output $OUTPUT_DIR $EXTRA_OPS $@"
```
**NB:** Use exactly the same Python version as your [prerequisites](#prereqs) "need" (only the major and minor version numbers e.g. 3.6, not 3.6.7).

<a name="use_run_local"></a>
#### Use `run_local.sh`

See [above](#official_docker) for how to specify [datasets](#datasets) and [models](#models).

##### Example: MobileNet non-quantized
```bash
$ ck install package --tags=mlperf,image-classification,model,tf,mobilenet,non-quantized
$ export MODEL_DIR=`ck locate env --tags=model,tf,mobilenet,non-quantized`
$ export DATA_DIR=`ck locate env --tags=dataset,imagenet,val`
$ export EXTRA_OPS="--count 1024 --scenario Offline"
$ ./run_local.sh tf mobilenet cpu
...
TestScenario.Offline qps=237.10, mean=3.3406, time=4.319, queries=1024, tiles=50.0:2.9683,80.0:4.2340,90.0:4.2692,95.0:4.2827,99.0:4.2932,99.9:4.2932
```

<a name="training_0_7"></a>
## MLPerf Training v0.7

**TODO**
