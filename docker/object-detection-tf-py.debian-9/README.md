# [MLPerf Inference - Object Detection - TF-Python (Debian 9)](https://hub.docker.com/r/ctuning/object-detection-tf-py.debian-9)

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
$ docker pull ctuning/object-detection-tf-py.debian-9
```

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:object-detection-tf-py.debian-9
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:object-detection-tf-py.debian-9`
$ docker build -f Dockerfile -t ctuning/object-detection-tf-py.debian-9 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Object Detection (default command)

##### Non-quantized, 50 images
```bash
$ ck run docker:object-detection-tf-py.debian-9
```
**NB:** Equivalent to:
```bash
$ docker run --rm ctuning/object-detection-tf-py.debian-9 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
Graph loaded in 0.923238s
All images loaded in 17.265170s
All images detected in 1.988970s
Average detection time: 0.040591s
mAP: 0.3148934914889957
Recall: 0.3225293342489256
--------------------------------
```

<a name="image_default_run_custom"></a>
#### Object Detection (custom command)

##### Non-quantized, 5000 images
```bash
$ docker run --rm ctuning/object-detection-tf-py.debian-9 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=5000 \
    "
...
Summary:
-------------------------------
Graph loaded in 0.937587s
All images loaded in 2006.936262s
All images detected in 272.221948s
Average detection time: 0.054455s
mAP: 0.23111107753357035
Recall: 0.26304841188725403
--------------------------------
```

##### Quantized, 50 images
```bash
$ docker run --rm ctuning/object-detection-tf-py.debian-9 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.weights=ssd-mobilenet,quantized \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=50 \
    "
...
Summary:
-------------------------------
Graph loaded in 1.092699s
All images loaded in 12.919672s
All images detected in 2.137745s
Average detection time: 0.043627s
mAP: 0.32625778039773207
Recall: 0.33433530428110675
--------------------------------
```

##### Quantized, 5000 images
```bash
$ docker run --rm ctuning/object-detection-tf-py.debian-9 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.weights=ssd-mobilenet,quantized \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=5000 \
    "
...
Summary:
-------------------------------
Graph loaded in 1.589762s
All images loaded in 1273.597364s
All images detected in 213.662603s
Average detection time: 0.042741s
mAP: 0.23594222525632427
Recall: 0.26864982712779556
--------------------------------
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/object-detection-tf-py.debian-9 bash
```
