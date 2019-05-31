# [Debian](https://hub.docker.com/_/debian/) 9

1. [Default image](#image_default) (9 latest)
    - [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Image Classification (default command)](#image_default_run_default)
        - [Image Classification (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:image-classification-tflite.debian-9
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:image-classification-tflite.debian-9`
$ docker build -f Dockerfile -t image-classification-tflite.debian-9 .
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
$ docker run --rm image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,non-quantized"
```

<a name="image_default_run_custom"></a>
#### Image Classification (custom command)
```bash
$ docker run --rm image-classification-tflite.debian-9 \
"ck run program:image-classification-tflite --dep_add_tags.weights=resnet,no-argmax --env.CK_BATCH_COUNT=10"
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm image-classification-tflite.debian-9 bash
```
