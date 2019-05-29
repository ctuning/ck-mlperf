# [Ubuntu](https://hub.docker.com/_/ubuntu/) 18.04

1. [Default image](#image_default)
    - [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Image Classification (default command)](#image_default_run_default)
        - [Image Classification (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)


**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

**NB:** We currently use the `ubuntu:18.04` base image.
You may wish to switch to a stable image e.g. `ubuntu:bionic-20190515`.

<a name="image_default"></a>
## Default image

<a name="image_default_build"></a>
### Build
```bash
$ cd `ck find docker:image-classification-tflite-ubuntu-18.04`
$ docker build -f Dockerfile -t image-classification-tflite.ubuntu-18.04 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Image Classification (default command)
```bash
$ docker run --rm image-classification-tflite.ubuntu-18.04
```
**NB:** Equivalent to:
```bash
$ docker run --rm image-classification-tflite.ubuntu-18.04 \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,non-quantized"
```

<a name="image_default_run_custom"></a>
#### Image Classification (custom command)
```bash
$ docker run --rm image-classification-tflite.ubuntu-18.04 \
"ck run program:image-classification-tflite --dep_add_tags.weights=resnet,no-argmax --env.CK_BATCH_COUNT=10"
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm image-classification-tflite.ubuntu-18.04 bash
```
