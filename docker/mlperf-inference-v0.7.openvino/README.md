# MLPerf Inference v0.7 - OpenVINO

This collection of images from [dividiti](http://dividiti.com) tests automated, customizable and reproducible [Collective Knowledge](http://cknowledge.org) workflows for OpenVINO workoads.

| `CK_TAG` (`Dockerfile`'s extension)  | Python | GCC   | Comments |
|-|-|-|-|
| `ubuntu-20.04` | 3.8.2 | 9.3.0 ||

<a name="setup_ck"></a>
## Set up Collective Knowledge

You will need to install [Collective Knowledge](http://cknowledge.org) to build images and save benchmarking results.
Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation) and then pull the ck-mlperf repository:

```bash
$ ck pull repo:ck-mlperf
```

**NB:** Refresh all CK repositories after any updates (e.g. bug fixes):
```bash
$ ck pull all
```

## Build

To build an image e.g. from `Dockerfile.ubuntu-20.04`:
```bash
$ export CK_IMAGE=mlperf-inference-v0.7.openvino CK_TAG=ubuntu-20.04
$ cd `ck find docker:$CK_IMAGE` && docker build -t ctuning/$CK_IMAGE:$CK_TAG -f Dockerfile.$CK_TAG .
```

## Run the default command

To run the default command of an image e.g. built from `Dockerfile.ubuntu-20.04`:
```bash
$ export CK_IMAGE=mlperf-inference-v0.7.openvino CK_TAG=ubuntu-20.04
$ docker run --rm ctuning/$CK_IMAGE:$CK_TAG
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
mAP=24.207%
```
