# [Centos](https://hub.docker.com/_/centos/) 7

1. [Default image](#image_default)
    - [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Image Classification (default command)](#image_default_run_default)
        - [Image Classification (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)

1. [Stable image](#image_stable)
    - [Build](#image_stable_build)
    - [Run](#image_stable_run)
        - [Image Classification (default command)](#image_stable_run_default)
        - [Image Classification (custom command)](#image_stable_run_custom)
        - [Bash](#image_stable_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:image-classification-tflite.centos-7
```
**NB:** Equivalent to:
```bash
$ docker build . -f Dockerfile -t image-classification-tflite.centos-7
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Image Classification (default command)
```bash
$ ck run docker:image-classification-tflite.centos-7
```
**NB:** Equivalent to:
```bash
$ docker run --rm image-classification-tflite.centos-7 \
"ck run program:image-classification-tflite"
```

<a name="image_default_run_custom"></a>
#### Image Classification (custom command)
```bash
$ docker run --rm image-classification-tflite.centos-7 \
"ck run program:image-classification-tflite --env.CK_BATCH_COUNT=10"
```

<a name="image_default_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm image-classification-tflite.centos-7 bash
```


<a name="image_stable"></a>
## Stable image

<a name="image_stable_build"></a>
### Build
```bash
$ docker build . -f Dockerfile.stable -t image-classification-tflite.centos-7.stable
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Image Classification (default command)
```bash
$ docker run --rm image-classification-tflite.centos-7.stable \
"ck run program:image-classification-tflite"
```

<a name="image_stable_run_custom"></a>
#### Image Classification (custom command)
```bash
$ docker run --rm image-classification-tflite.centos-7.stable \
"ck run program:image-classification-tflite --env.CK_BATCH_COUNT=10"
```

<a name="image_stable_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm image-classification-tflite.centos-7.stable bash
```
