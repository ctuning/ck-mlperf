# MLPerf Inference - Object Detection - TensorFlow with Intel MKL

[This image](https://hub.docker.com/r/ctuning/mlperf-inference-vision-with-ck.intel.ubuntu-18.04) is based on
[the MKL-optimized TensorFlow image](https://hub.docker.com/r/intelaipg/intel-optimized-tensorflow/) from Intel
(which is in turn based on Ubuntu 18.04).

The image includes about a dozen of [TensorFlow models for object detection](#models), the [COCO 2017 validation dataset](http://cocodataset.org),
and MKL-optimized [TensorFlow 1.15.2](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.2).

1. [Setup](#setup)
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

Note that you may need to run commands below with `sudo`, unless you [manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="setup_ck"></a>
## Set up Collective Knowledge

You will need to install [Collective Knowledge](http://cknowledge.org) to build images and save benchmarking results.
Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation) and then pull our object detection repository:

```bash
$ ck pull repo:ck-mlperf
```

**NB:** Refresh all CK repositories after any updates (e.g. bug fixes):
```bash
$ ck pull all
```
(This only updates CK repositories on the host system. To update the Docker image, [rebuild](#build) it using the `--no-cache` flag.)

### Set up environment variables

Set up the variable to contain the image name:
```bash
$ export CK_IMAGE=mlperf-inference-vision-with-ck.intel.ubuntu-18.04
```

Set up the variable that points to the directory that contains your CK repositories (usually `~/CK` or `~/CK_REPOS`):
```bash
$ export CK_REPOS=${HOME}/CK
```

<a name="image_download"></a>
## Download from Docker Hub

To download a prebuilt image from Docker Hub, run:
```
$ docker pull ctuning/${CK_IMAGE}
```

**NB:** As the prebuilt TensorFlow variant does not support AVX2 instructions, we advise to use the TensorFlow variant built from sources on compatible hardware.
In fact, as the prebuilt image was built on an [HP Z640 workstation](http://h20195.www2.hp.com/v2/default.aspx?cc=ie&lc=en&oid=7528701)
with an [Intel(R) Xeon(R) CPU E5-2650 v3](https://ark.intel.com/products/81705/Intel-Xeon-Processor-E5-2650-v3-25M-Cache-2_30-GHz) (launched in Q3'14), we advise
to [rebuild](#build) the image on your system.

<a name="image_build"></a>
## Build

To build an image on your system, run:
```bash
$ ck build docker:${CK_IMAGE}
```

**NB:** This CK command is equivalent to:
```bash
$ cd `ck find docker:${CK_IMAGE}`
$ docker build --no-cache -f Dockerfile -t ctuning/${CK_IMAGE} .
```

<a name="usage"></a>
# Usage

<a name="run"></a>
## Run inference once

Once you have downloaded or built an image, you can run inference on the CPU as follows:
```bash
$ docker run --env-file ${CK_REPOS}/ck-mlperf/docker/${CK_IMAGE}/env.list --rm ctuning/${CK_IMAGE} \
        "ck run program:mlperf-inference-vision --cmd_key=direct \
        --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
        --env.CK_METRIC_TYPE=COCO \
        --env.CK_LOADGEN_SCENARIO=SingleStream \
        --env.CK_LOADGEN_MODE='--accuracy' \
        --dep_add_tags.weights=ssd,mobilenet-v1,quantized,mlperf,tf \
        --dep_add_tags.lib-tensorflow=vcpu --env.CUDA_VISIBLE_DEVICES=-1 --env.CK_LOADGEN_BACKEND=tensorflow \
        --env.CK_LOADGEN_REF_PROFILE=default_tf_object_det_zoo \
        --skip_print_timers"
```
Here, we run inference on 50 images on the CPU using the quantized SSD-MobileNet model.

**NB:** This is equivalent to the default run command:
```bash
$ docker run --rm ctuning/$CK_IMAGE
```

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
| `--env.CK_LOADGEN_BACKEND`  | tensorflow | tensorflow |  |
| `--env.CK_LOADGEN_REF_PROFILE` | mobilenet-tf,default_tf_object_det_zoo,default_tf_trt_object_det_zoo,tf_yolo,tf_yolo_trt | mobilenet-tf | The "LoadGen profile" - combines aspects of model and backend |
| `--env.CK_LOADGEN_SCENARIO` | SingleStream,Offline,MultiStream | SingleStream | The LoadGen testing scenario |
| `--env.CK_LOADGEN_MODE` | "--accuracy","" | "--accuracy" | LoadGen mode - empty line stands for Performance mode |


<a name="benchmark"></a>
## Benchmark models individually

When you run inference using `ck run`, the results get printed but not saved (and some won't be even printed).
You can use `ck benchmark` to save the results on the host system as CK experiment entries (JSON files).

Let's set up a variable that points to the directory on the host computer where you want to collect the experiments from,
making sure $USER has write access to it:

```bash
$ export CK_EXPERIMENTS_DIR=/data/$USER/mlperf-inference-vision-experiments

$ mkdir -p ${CK_EXPERIMENTS_DIR}
```

When running `ck benchmark` via Docker, we map the internal output directory to `$CK_EXPERIMENTS_DIR` on the host
in order to access the results easier (using parameters for a custom Yolo v3 model for a change) :

```bash
$ docker run --env-file ${CK_REPOS}/ck-mlperf/docker/${CK_IMAGE}/env.list \
        --user=$(id -u):1500 --volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
        --rm ctuning/${CK_IMAGE} \
        "ck benchmark program:mlperf-inference-vision --cmd_key=direct --repetitions=1 \
        --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
        --env.CK_METRIC_TYPE=COCO \
        --env.CK_LOADGEN_SCENARIO=SingleStream \
        --env.CK_LOADGEN_MODE='--accuracy' \
        --dep_add_tags.weights=yolo-v3 \
        --dep_add_tags.lib-tensorflow=vcpu \
        --env.CK_LOADGEN_BACKEND=tensorflow \
        --env.CK_LOADGEN_REF_PROFILE=tf_yolo_trt \
        --record --record_repo=local \
        --record_uoa=mlperf.open.object-detection.cpu.yolo_v3_coco.singlestream.accuracy \
        --tags=mlperf,open,object-detection,cpu,yolo_v3_coco,singlestream,accuracy \
        --skip_print_timers --skip_stat_analysis --process_multi_keys"
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
- `--repetitions`: the number of times to run an experiment (3 by default); we typically use `--repetitions=1` for experiments that measure accuracy and e.g. `--repetitions=10` for experiments that measure performance.
- `--record`, `--record_repo=local`: must be present to have the results saved in the mounted volume.
- `--record_uoa`: a unique name for each CK experiment entry; here, `mlperf.open.object-detection` is the common prefix for all experiments, `cpu` is the TensorFlow backend, `ssd-mobilenet-quantized` is unique for each model, `accuracy` indicates the accuracy mode.
- `--tags`: specify the comma-separated tags for each CK experiment entry; we typically use parts of the experiment entry name.


<a name="explore"></a>
## Explore design space

Putting this all together, we provide a shell script which can be found under:
```
$ ck find script:mlperf-inference-v0.5.open.object-detection
```

The script launches full design space explorationo ver all the object detection [models](#models) and available TensorFlow backends.


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
