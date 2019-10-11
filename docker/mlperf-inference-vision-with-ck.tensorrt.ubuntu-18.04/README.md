# FIXME: Currently identical to: Object Detection TensorFlow (Python) Docker image

[This image](https://hub.docker.com/r/ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04) is based on
[the TensorRT 19.07 image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_19-07.html) from NVIDIA
(which is in turn based on Ubuntu 18.04) with [CUDA](https://developer.nvidia.com/cuda-zone) 10.1 and [TensorRT](https://developer.nvidia.com/tensorrt) 5.1.5.

The image includes about a dozen of [TensorFlow models for object detection](#models), the [COCO 2017 validation dataset](http://cocodataset.org),
and two [TensorFlow 1.14.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.14.0) variants:
- TensorFlow prebuilt for the CPU (installed via pip).
- TensorFlow built from sources for the GPU, with TensorRT support enabled.

**NB:** The latter variant can be forced to run on the CPU. We used to have two
separate Docker images based on Ubuntu 18.04 to measure the performance of
prebuilt TensorFlow vs TensorFlow built from sources on the CPU, but it is
easier to manage a single image.

1. [Setup](#setup)
    - [Set up NVIDIA Docker](#setup_nvidia)
    - [Set up Collective Knowledge](#setup_ck)
    - [Download](#image_download) and/or [Build](#image_build) images
1. [Usage](#usage)
    - [Run once](#run)
        - [Models](#models)
        - [Flags](#flags)
    - [Benchmark](#benchmark)
        - [Docker parameters](#parameters_docker)
        - [CK parameters](#parameters_ck)
    - [Explore](#explore)
    - [Analyze](#analyze)

<a name="setup"></a>
# Setup

<a name="setup_nvidia"></a>
## Set up NVIDIA Docker

As our GPU image is based on [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), please follow instructions there to set up your system.

Note that you may need to run commands below with `sudo`, unless you [manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="setup_ck"></a>
## Set up Collective Knowledge

You will need to install [Collective Knowledge](http://cknowledge.org) to build images and save benchmarking results.
Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation) and then pull our object detection repository:

```bash
$ ck pull repo:ck-object-detection
```

**NB:** Refresh all CK repositories after any updates (e.g. bug fixes):
```bash
$ ck pull all
```
(This only updates CK repositories on the host system. To update the Docker image, [rebuild](#build) it using the `--no-cache` flag.)

<a name="image_download"></a>
## Download from Docker Hub

To download a prebuilt image from Docker Hub, run:
```
$ docker pull ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04
```

**NB:** As the prebuilt TensorFlow variant does not support AVX2 instructions, we advise to use the TensorFlow variant built from sources on compatible hardware.
In fact, as the prebuilt image was built on an [HP Z640 workstation](http://h20195.www2.hp.com/v2/default.aspx?cc=ie&lc=en&oid=7528701)
with an [Intel(R) Xeon(R) CPU E5-2650 v3](https://ark.intel.com/products/81705/Intel-Xeon-Processor-E5-2650-v3-25M-Cache-2_30-GHz) (launched in Q3'14), we advise
to [rebuild](#build) the image on your system.

<a name="image_build"></a>
## Build

To build an image on your system, run:
```bash
$ ck build docker:object-detection-tf-py.tensorrt.ubuntu-18.04
```

**NB:** This CK command is equivalent to:
```bash
$ cd `ck find docker:object-detection-tf-py.tensorrt.ubuntu-18.04`
$ docker build --no-cache -f Dockerfile -t ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04 .
```

<a name="usage"></a>
# Usage

<a name="run"></a>
## Run inference once

Once you have downloaded or built an image, you can run inference on the CPU e.g. as follows:
```bash
$ docker run --rm ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.lib-tensorflow=vprebuilt \
        --dep_add_tags.weights=ssd-mobilenet,quantized \
        --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50 \
    "
```
Here, we run inference on 50 images on the CPU using the quantized SSD-MobileNet model.

**NB:** This is equivalent to the default run command:
```bash
$ ck run docker:object-detection-tf-py.tensorrt.ubuntu-18.04
```

To run inference on the GPU, add the `--runtime=nvidia` flag:

```bash
$ docker run --runtime=nvidia --rm ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.lib-tensorflow=vsrc \
        --dep_add_tags.weights=ssd-mobilenet,quantized \
        --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50 \
        --env.CK_ENABLE_TENSORRT=1 \
        --env.CK_TENSORRT_DYNAMIC=1 \
    "
```
Here, we additionally request to use TensorRT in the [dynamic mode](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#static-dynamic-mode).

We describe all supported [models](#models) and [flags](#flags) below.

<a name="models"></a>
### Models

Our [TensorFlow-Python application](https://github.com/ctuning/ck-tensorflow/blob/master/program/object-detection-tf-py/README.md) supports the following TensorFlow models trained on the COCO 2017 dataset. With the exception of a [TensorFlow reimplementation of YOLO v3](https://github.com/YunYang1994/tensorflow-yolov3), all the models come from the [TensorFlow Object Detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
Note that we report the accuracy reference (mAP in %) on the COCO 2017 validation dataset (5,000 images).

| Model | Unique CK Tags (`<tags>`) | Is Custom? | mAP in % |
| --- | --- | --- | --- |
| [`faster_rcnn_nas_lowproposals_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)       | `rcnn,nas,lowproposals,vcoco`     | 0 | 44.340195 |
| [`faster_rcnn_resnet50_lowproposals_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  | `rcnn,resnet50,lowproposals`      | 0 | 24.241037 |
| [`faster_rcnn_resnet101_lowproposals_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `rcnn,resnet101,lowproposals`     | 0 | 32.594327 |
| [`faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `rcnn,inception-resnet-v2,lowproposals` | 0 | 36.520117 |
| [`faster_rcnn_inception_v2_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)           | `rcnn,inception-v2`               | 0 | 28.309626 |
| [`ssd_inception_v2_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)          | `ssd,inception-v2`                         | 0 | 27.765988 |
| [`ssd_mobilenet_v1_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)          | `ssd,mobilenet-v1,non-quantized,mlperf,tf` | 0 | 23.111170 |
| [`ssd_mobilenet_v1_quantized_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)| `ssd,mobilenet-v1,quantized,mlperf,tf`     | 0 | 23.591693 |
| [`ssd_mobilenet_v1_fpn_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)      | `ssd,mobilenet-v1,fpn`                     | 0 | 35.353170 |
| [`ssd_resnet_50_fpn_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)         | `ssd,resnet50,fpn`                         | 0 | 38.341120 |
| [`ssdlite_mobilenet_v2_coco`](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)      | `ssdlite,mobilenet-v2,vcoco`               | 0 | 24.281540 |
| [`yolo_v3_coco`](https://github.com/YunYang1994/tensorflow-yolov3)                                                                          | `yolo-v3`                                  | 1 | 28.532508 |

Each model can be selected by adding the `--dep_add_tags.weights=<tags>` flag when running a customized command for the container.
For example, to run inference on the quantized SSD-MobileNet model, add `--dep_add_tags.weights=ssd-mobilenet,quantized`; to run inference on the YOLO model, add `--dep_add_tags.weights=yolo`; and so on.

<a name="flags"></a>
### Flags

| Env Flag                    | Possible Values  | Default Value | Description |
|-|-|-|-|
| `--env.CK_CUSTOM_MODEL`     | 0,1              | 0 | Specifies whether the model comes from the TensorFlow zoo (`0`) or from another source (`1`). (Models from other sources have to implement their own preprocess, postprocess and get tensor functions, as explained in the [application documentation](https://github.com/ctuning/ck-tensorflow/blob/master/program/object-detection-tf-py/README.md#support-for-custom-models).) |
| `--env.CK_ENABLE_BATCH`     | 0,1              | 0 | Specifies whether batching should be enabled (must be used for `--env.CK_BATCH_SIZE` to take effect). |
| `--env.CK_BATCH_SIZE`       | positive integer | 1 | Specifies the number of images to process in a single batch (if not `1`, must be used with `--env.CK_ENABLE_BATCH=1`). |
| `--env.CK_BATCH_COUNT`      | positive integer | 1 | Specifies the number of batches to be processed. |
| `--env.CK_ENV_IMAGE_WIDTH`, `--env.CK_ENV_IMAGE_HEIGHT` | positive integer | Model-specific (set by CK) | These parameters can be used to resize at runtime the input images to a different size than the default size for the model. (This usually decreases the accuracy.) |
| `--env.CK_ENABLE_TENSORRT`  | 0,1              | 0 | Enables the TensorRT backend (only to be used with TensorFlow built from sources). |
| `--env.CK_TENSORRT_DYNAMIC` | 0,1              | 0 | Enables the [TensorRT dynamic mode](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#static-dynamic-mode) (must be used with `--env.CK_ENABLE_TENSORRT=1`). |
| `--env.CUDA_VISIBLE_DEVICES`| list of integers | N/A | Specifies which GPUs should be used by TensorFlow; `-1` forces TensorFlow to use the CPU (even with TensorFlow built from sources). |


<a name="benchmark"></a>
## Benchmark models individually

When you run inference using `ck run`, the result gets printed but not saved.
You can use `ck benchmark` to save the result on the host system as CK experiment entries (JSON files) e.g. as follows:

```bash
$ docker run --runtime=nvidia \
    --env-file `ck find docker:object-detection-tf-py.tensorrt.ubuntu-18.04`/env.list \
    --volume=<folder_for_results>:/home/dvdt/CK_REPOS/local/experiment \
    --user=$(id -u):1500 \
    --rm ctuning/object-detection-tf-py.tensorrt.ubuntu-18.04 \
    "ck benchmark program:object-detection-tf-py \
        --dep_add_tags.lib-tensorflow=vsrc \
        --dep_add_tags.weights=ssd-mobilenet,quantized \
        --env.CK_BATCH_COUNT=50 \
        --repetitions=1 \
        --record \
        --record_repo=local \
        --record_uoa=object-detection-tf-py-ssd-mobilenet-quantized-accuracy \
        --tags=object-detection,tf-py,ssd-mobilenet,quantized,accuracy \
    "
```

<a name="parameters_docker"></a>
### Docker parameters

- `--env-file`: the path to the `env.list` file, which is usually located in the same folder as the Dockerfile. (Currently, the `env.list` files are identical for all the images.)
- `--volume`: a folder with read/write permissions for the user that serves as shared space ("volume") between the host and the container.
- `--user`: your user id on the host system and a fixed group id (1500) needed to access files in the container.

#### Gory details

We ask you to launch a container with `--user=$(id -u):1500`, where `$(id -u)` gives your
user id on the host system and `1500` is the fixed group id of the `dvdtg` group in the image.
We also ask you to mount a folder with read/write permissions with `--volume=<folder_for_results>`.
This folder gets mapped to the `/home/dvdt/CK_REPOS/local/experiment` folder in the image.
While the `experiment` folder belongs to the `dvdt` user, it is made accessible to the `dvdtg` group.
Therefore, you can retrieve the results of a container run under your user id from this folder.


<a name="parameters_ck"></a>
### CK parameters

- `--dep_add_tags.lib-tensorflow`: specify `vsrc` to use TensorFlow built from sources controlling its execution via [flags](#flags) and `vprebuilt` to use prebuilt TensorFlow on the CPU.
- `--dep_add_tags.weights`: specify the tags for a particular [model](#models).
- `--env.CK_BATCH_COUNT`: the number of batches to be processed; assuming `--env.CK_BATCH_SIZE=1`, we typically use `--env.CK_BATCH_COUNT=5000` for experiments that measure accuracy over the COCO 2017 validation set and e.g. `--env.CK_BATCH_COUNT=2` for experiments that measure performance. (With TensorFlow, the first batch is usually slower than all subsequent batches. Therefore, its execution time has to be discarded. The execution times of subsequent batches will be averaged.)
- `--repetitions`: the number of times to run an experiment (3 by default); we typically use `--repetitions=1` for experiments that measure accuracy and e.g. `--repetitions=10` for experiments that measure performance.
- `--record`, `--record_repo=local`: must be present to have the results saved in the mounted volume.
- `--record_uoa`: a unique name for each CK experiment entry; here, `object-detection-tf-py` (the name of the program) is the same for all experiments, `ssd-mobilenet-quantized` is unique for each model, `accuracy` indicates the accuracy mode.
- `--tags`: specify the tags for each CK experiment entry; we typically make them similar to the experiment entry name.


<a name="explore"></a>
## Explore design space

Putting this all together, we provide two shell scripts that launch full design space exploration
in the accuracy mode (`--repetitions=1`) and the performance mode (`--repetitions=10`)
with the corresponding experiment names and tags:
- prebuilt CPU (no AVX2 FMA), CPU, CUDA, TensorRT with the dynamic mode disabled, TensorRT with the dynamic mode enabled.
- over all the object detection [models](#models).
- in the performance mode, over several batch sizes (1, 2, 4, 8, 16).

The scripts can be found under:
```
$ ck find script:<script_name>
```
where `<script_name>` is either 'dse-acc' or 'dse-perf'

To use the script, you have to modify the first lines in order to adapt the path to the your host system.
You can also modify other parameters, like the list of models to test or the batch sizes and counts.


<a name="analyze"></a>
## Analyze the results

### Copy the results to a machine for analysis

Once you have accumulated some experiment entries in `<folder_for_results>`, you can zip them:
```bash
$ cd <folder_for_results>
$ zip -rv <file_with_results>.zip {.cm,*}
```
copy `<file_with_results>.zip` to a machine where you would like to analyze them,
create there a new repository with a placeholder for experiment entries:
```bash
$ ck add repo:object-detection-tf-py-experiments --quiet
$ ck add object-detection-tf-py-experiments:experiment:dummy --common_func
$ ck rm  object-detection-tf-py-experiments:experiment:dummy --force
```
or:
```bash
$ ck add repo:object-detection-tf-py-experiments --quiet
$ ck create_entry --data_uoa=experiment --data_uid=bc0409fb61f0aa82 \
--path=`ck find repo:object-detection-tf-py-experiments`
```
and, finally, extract the results:
```
$ unzip <file_with_result>.zip -d `ck find repo:object-detection-tf-py-experiments`/experiment
$ ck list object-detection-tf-py-experiments:experiment:*
...
```

### Visualize the results via Jupyter

**TBC**
