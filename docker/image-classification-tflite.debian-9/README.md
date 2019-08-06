# [MLPerf Inference - Image Classification - TFLite (Debian 9)](https://hub.docker.com/r/ctuning/image-classification-tflite.debian-9)

1. [Default image](#image_default) (based on [Debian](https://hub.docker.com/_/debian/) 9 latest)
    - [Download](#image_default_download) or [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Image Classification (default command)](#image_default_run_default)
        - [Image Classification (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_download"></a>
### Download
```
$ docker pull ctuning/image-classification-tflite.debian-9
```

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:image-classification-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:image-classification-tflite.debian-9`
$ docker build -f Dockerfile -t ctuning/image-classification-tflite.debian-9 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Image Classification (default command)
```bash
$ ck run docker:image-classification-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ docker run --rm ctuning/image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,non-quantized --env.CK_BATCH_COUNT=2"
```

<a name="image_default_run_custom"></a>
#### Image Classification (custom command)
##### ResNet
```bash
$ docker run --rm ctuning/image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=resnet,no-argmax --env.CK_BATCH_COUNT=500"
...
Summary:
-------------------------------
Graph loaded in 0.001373s
All images loaded in 1.173109s
All images classified in 83.794106s
Average classification time: 0.167370s
Accuracy top 1: 0.762 (381 of 500)
Accuracy top 5: 0.93 (465 of 500)
--------------------------------
```
##### MobileNet non-quantized
```bash
$ docker run --rm ctuning/image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,non-quantized --env.CK_BATCH_COUNT=500"
...
Summary:
-------------------------------
Graph loaded in 0.001232s
All images loaded in 0.688655s
All images classified in 27.330872s
Average classification time: 0.054614s
Accuracy top 1: 0.724 (362 of 500)
Accuracy top 5: 0.896 (448 of 500)
--------------------------------
```
##### MobileNet quantized
```bash
$ docker run --rm ctuning/image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,quantized --env.CK_BATCH_COUNT=500"
...
Summary:
-------------------------------
Graph loaded in 0.001149s
All images loaded in 0.032835s
All images classified in 81.675827s
Average classification time: 0.163378s
Accuracy top 1: 0.728 (364 of 500)
Accuracy top 5: 0.898 (449 of 500)
--------------------------------
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/image-classification-tflite.debian-9 bash
```
