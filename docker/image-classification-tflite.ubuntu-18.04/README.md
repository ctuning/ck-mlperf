# [Ubuntu](https://hub.docker.com/_/ubuntu/) 18.04

1. [Default image](#image_default) (18.04 latest)
    - [Build](#image_default_build)
    - [Run](#image_default_run)
        - [Image Classification (default command)](#image_default_run_default)
        - [Image Classification (custom command)](#image_default_run_custom)
        - [Bash](#image_default_run_bash)
1. [Dashboard image](#image_dashboard) (18.04 latest)
    - [Build](#image_dashboard_build)
    - [Run](#image_dashboard_run)
        - [Interactive dashboard](#image_dashboard_run_dashboard)
        - [Image Classification (default command)](#image_dashboard_run_default)
        - [Image Classification (custom command)](#image_dashboard_run_custom)
        - [Bash](#image_dashboard_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_default"></a>
## Default image

<a name="image_default_build"></a>
### Build
```bash
$ ck build docker:image-classification-tflite.ubuntu-18.04
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:image-classification-tflite.ubuntu-18.04`
$ docker build -f Dockerfile -t image-classification-tflite.ubuntu-18.04 .
```

<a name="image_default_run"></a>
### Run

<a name="image_default_run_default"></a>
#### Image Classification (default command)
```bash
$ ck run docker:image-classification-tflite.ubuntu-18.04
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


<a name="image_dashboard"></a>
## Dashboard image

<a name="image_dashboard_build"></a>
### Build
```bash
$ cd `ck find docker:image-classification-tflite.ubuntu-18.04`
$ docker build -f Dockerfile.dashboard -t image-classification-tflite.ubuntu-18.04.dashboard .
```

<a name="image_dashboard_run"></a>
### Run

<a name="image_dashboard_run_dashboard"></a>
#### Dashboard
```bash
$ docker run -it --publish 3355:3344 --rm image-classification-tflite.ubuntu-18.04.dashboard
```

<a name="image_dashboard_run_default"></a>
#### Image Classification (default command)
```bash
$ docker run --rm image-classification-tflite.ubuntu-18.04.dashboard \
"ck run program:image-classification-tflite --dep_add_tags.weights=mobilenet,non-quantized"
```

<a name="image_dashboard_run_custom"></a>
#### Image Classification (custom command)
```bash
$ docker run --rm image-classification-tflite.ubuntu-18.04.dashboard \
"ck run program:image-classification-tflite --dep_add_tags.weights=resnet,no-argmax --env.CK_BATCH_COUNT=10"
```

<a name="image_dashboard_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm image-classification-tflite.ubuntu-18.04.dashboard bash
```
