# [MLPerf Inference - Object Detection - TFLite (Debian 9)](https://hub.docker.com/r/ctuning/object-detection-tflite.debian-9)

1. [Default image](#image_default) (based on [Debian](https://hub.docker.com/_/debian/) 9 latest)
    - [Download](#image_default_download) or [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Object Detection (default command)](#image_default_run_default)
        - [Object Detection (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_download"></a>
### Download
```
$ docker pull ctuning/object-detection-tflite.debian-9
```

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:object-detection-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:object-detection-tflite.debian-9`
$ docker build -f Dockerfile -t ctuning/object-detection-tflite.debian-9 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Object Detection (default command)

##### Regular NMS; 50 images
```bash
$ ck run docker:object-detection-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ docker run --rm ctuning/object-detection-tflite.debian-9 \
    "ck run program:object-detection-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=regular \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
All images loaded in 0.147986s
Average image load time: 0.002960s
All images detected in 5.917367s
Average detection time: 0.117556s
mAP: 0.29672520317694373
Recall: 0.3050474339529269
--------------------------------
```

<a name="image_default_run_custom"></a>
#### Object Detection (custom command)

##### Fast NMS; 50 images
```bash
$ docker run --rm ctuning/object-detection-tflite.debian-9 \
    "ck run program:object-detection-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=fast \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
All images loaded in 0.146889s
Average image load time: 0.002938s
All images detected in 5.868071s
Average detection time: 0.116611s
mAP: 0.29624782705876884
Recall: 0.30501085304815917
--------------------------------
```

##### Regular NMS; 5000 images
```bash
$ docker run --rm ctuning/object-detection-tflite.debian-9 \
    "ck run program:object-detection-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=regular \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=5000 \
    "
...
Summary:
-------------------------------
All images loaded in 14.741492s
Average image load time: 0.002948s
All images detected in 587.250183s
Average detection time: 0.117443s
mAP: 0.22349680978666922
Recall: 0.2550505369422975
--------------------------------
```

##### Fast NMS; 5000 images
```bash
$ docker run --rm ctuning/object-detection-tflite.debian-9 \
    "ck run program:object-detection-tflite \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=fast \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=5000 \
    "
...
Summary:
-------------------------------
All images loaded in 14.953116s
Average image load time: 0.002991s
All images detected in 587.7276s
Average detection time: 0.117538s
mAP: 0.21859688835124763
Recall: 0.24801510024502602
--------------------------------
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/object-detection-tflite.debian-9 bash
```
