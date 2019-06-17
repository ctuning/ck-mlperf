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

##### 50 images
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
```

<a name="image_default_run_custom"></a>
#### Object Detection (custom command)

##### 5000 images
```bash
$ docker run --rm ctuning/object-detection-tf-py.debian-9 \
    "ck run program:object-detection-tf-py \
        --dep_add_tags.weights=ssd-mobilenet,non-quantized \
        --dep_add_tags.dataset=coco.2017,full --env.CK_BATCH_COUNT=5000 \
    "
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/object-detection-tf-py.debian-9 bash
```
