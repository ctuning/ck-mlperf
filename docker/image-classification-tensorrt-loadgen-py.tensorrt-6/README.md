# MLPerf Inference - Image Classification - NVIDIA TensorRT

[This image]() is based on
[the TensorRT 19.12 image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/rel_19-12.html) from NVIDIA
(which is in turn based on Ubuntu 18.04) with [CUDA](https://developer.nvidia.com/cuda-zone) 10.2 and [TensorRT](https://developer.nvidia.com/tensorrt) 6.0.1.

1. [Setup](#setup)
    - [Set up NVIDIA Docker](#setup_nvidia)
    - [Set up Collective Knowledge](#setup_ck)
    - [Download](#image_download) and/or [Build](#image_build) images
1. [Usage](#usage)
    - [Run once](#run)
    - [Benchmark](#benchmark)
        - [Docker parameters](#parameters_docker)
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
$ export IMAGE=image-classification-tensorrt-loadgen-py.tensorrt-6
```

Set up the variable that points to the directory that contains your CK repositories (usually `~/CK` or `~/CK_REPOS`):
```bash
$ export CK_REPOS=${HOME}/CK
```

<a name="image_download"></a>
## Download from Docker Hub

To download a prebuilt image from Docker Hub, run:
```
$ docker pull ctuning/${IMAGE}
```

<a name="image_build"></a>
## Build

To build an image on your system, run:
```bash
$ ck build docker:${IMAGE}
```

**NB:** This CK command is equivalent to:
```bash
$ cd `ck find docker:${IMAGE}`
$ docker build --no-cache -f Dockerfile -t ctuning/${IMAGE} .
```

<a name="usage"></a>
# Usage

<a name="run"></a>
## Run inference once

Once you have downloaded or built an image, you can run inference in the accuracy or performance mode as follows.

### Accuracy mode

```bash
$ docker run --runtime=nvidia --env-file ${CK_REPOS}/ck-mlperf/docker/${IMAGE}/env.list --rm ctuning/${IMAGE} \
  "ck run program:image-classification-tensorrt-loadgen-py --skip_print_timers --env.CK_SILENT_MODE \
  --env.CK_LOADGEN_MODE=AccuracyOnly --env.CK_LOADGEN_SCENARIO=MultiStream \
  --env.CK_LOADGEN_MULTISTREAMNESS=32 --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_DATASET_SIZE=500 --env.CK_LOADGEN_BUFFER_SIZE=500 \
  --env.CK_LOADGEN_CONF_FILE=/home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/user.conf \
  --dep_add_tags.weights=model,tensorrt,resnet,converted-from-onnx,fp32,maxbatch.32 \
  && echo '--------------------------------------------------------------------------------' \
  && echo 'mlperf_log_summary.txt' \
  && echo '--------------------------------------------------------------------------------' \
  && cat  /home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/tmp/mlperf_log_summary.txt \
  && echo '--------------------------------------------------------------------------------' \
  && echo 'mlperf_log_detail.txt' \
  && echo '--------------------------------------------------------------------------------' \
  && cat  /home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/tmp/mlperf_log_detail.txt \
  && echo ''"
...
--------------------------------
accuracy=75.200%, good=376, total=500

--------------------------------
...
--------------------------------------------------------------------------------
mlperf_log_summary.txt
--------------------------------------------------------------------------------

No warnings encountered during test.

No errors encountered during test.
```

#### Performance mode

```bash
$ docker run --runtime=nvidia --env-file ${CK_REPOS}/ck-mlperf/docker/${IMAGE}/env.list --rm ctuning/${IMAGE} \
  "ck run program:image-classification-tensorrt-loadgen-py --skip_print_timers --env.CK_SILENT_MODE \
  --env.CK_LOADGEN_MODE=PerformanceOnly --env.CK_LOADGEN_SCENARIO=MultiStream \
  --env.CK_LOADGEN_MULTISTREAMNESS=32 --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_COUNT_OVERRIDE=1440 \
  --env.CK_LOADGEN_DATASET_SIZE=500 --env.CK_LOADGEN_BUFFER_SIZE=1024 \
  --env.CK_LOADGEN_CONF_FILE=/home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/user.conf \
  --dep_add_tags.weights=model,tensorrt,resnet,converted-from-onnx,fp32,maxbatch.32 \
  && echo '--------------------------------------------------------------------------------' \
  && echo 'mlperf_log_summary.txt' \
  && echo '--------------------------------------------------------------------------------' \
  && cat  /home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/tmp/mlperf_log_summary.txt \
  && echo '--------------------------------------------------------------------------------' \
  && echo 'mlperf_log_detail.txt' \
  && echo '--------------------------------------------------------------------------------' \
  && cat  /home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/tmp/mlperf_log_detail.txt \
  && echo ''"
...
--------------------------------------------------------------------
|                LATENCIES (in nanoseconds and fps)                |
--------------------------------------------------------------------
Number of queries run:           46080
Min latency:                  47174403 ns   (21.198 fps)
Median latency:               52104957 ns   (19.192 fps)
Average latency:              51209991 ns   (19.527 fps)
90 percentile latency:        53049435 ns   (18.850 fps)
Max latency:                  56571231 ns   (17.677 fps)
--------------------------------------------------------------------
...
--------------------------------------------------------------------------------
mlperf_log_summary.txt
--------------------------------------------------------------------------------
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Multi Stream
Mode     : Performance
Samples per query : 32
Result is : INVALID
  Performance constraints satisfied : NO
  Min duration satisfied : Yes
  Min queries satisfied : Yes
Recommendations:
 * Reduce samples per query to improve latency.
...
```

Here, we run inference on 500 images using a TensorRT plan converted on-the-fly from the reference ResNet ONNX model.

**NB:** This is equivalent to the default run command:
```bash
$ docker run --rm ctuning/$IMAGE
```

In this example (on the NVIDIA GTX1080), the 99th percentile latency exceeds 50 ms in the MultiStream scenario,
which unfortunately makes the performance run **INVALID** according to the
[MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc#41-benchmarks) for the ResNet workload.


<a name="benchmark"></a>
## Benchmark with parameters

When you run inference using `ck run`, the results get printed but not saved.
You can use `ck benchmark` to save the results on the host system as CK experiment entries (JSON files).

Create a directory on the host computer where you want to store experiment entries e.g.:
```
```bash
$ export EXPERIMENTS_DIR=/data/$USER/tensorrt-experiments
$ mkdir -p ${EXPERIMENTS_DIR}
```
(**NB:** `USER` must have write access to this directory.)

When running `ck benchmark` via Docker, map the internal output directory to `$EXPERIMENTS_DIR` on the host:

```bash
$ export NUM_STREAMS=30
$ docker run --runtime=nvidia --env-file ${CK_REPOS}/ck-mlperf/docker/${IMAGE}/env.list \
  --user=$(id -u):1500 --volume ${EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment --rm ctuning/${IMAGE} \
  "ck benchmark program:image-classification-tensorrt-loadgen-py --repetitions=1 --env.CK_SILENT_MODE \
  --env.CK_LOADGEN_MODE=PerformanceOnly --env.CK_LOADGEN_SCENARIO=MultiStream \
  --env.CK_LOADGEN_MULTISTREAMNESS=${NUM_STREAMS} --env.CK_BATCH_SIZE=${NUM_STREAMS} \
  --env.CK_LOADGEN_COUNT_OVERRIDE=1440 \
  --env.CK_LOADGEN_DATASET_SIZE=500 --env.CK_LOADGEN_BUFFER_SIZE=1024 \
  --env.CK_LOADGEN_CONF_FILE=/home/dvdt/CK_REPOS/ck-mlperf/program/image-classification-tensorrt-loadgen-py/user.conf \
  --dep_add_tags.weights=model,tensorrt,resnet,converted-from-onnx,fp32,maxbatch.${NUM_STREAMS} \
  --record --record_repo=local \
  --record_uoa=mlperf.closed.image-classification.tensorrt.resnet.multistream.performance \
  --tags=mlperf,closed,image-classification,tensorrt,resnet,multistream,performance \
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

<a name="explore"></a>
## Explore

<a name="analyze"></a>
## Analyze

