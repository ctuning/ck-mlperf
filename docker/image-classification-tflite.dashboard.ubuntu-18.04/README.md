# [MLPerf Inference - Image Classification - TFLite (Ubuntu 18.04)](https://hub.docker.com/r/ctuning/image-classification-tflite.dashboard.ubuntu-18.04): Dashboard

1. [Dashboard image](#image_dashboard) (based on [Ubuntu](https://hub.docker.com/_/ubuntu/) 18.04 latest)
    - [Download](#image_dashboard_download) or [Build](#image_dashboard_build)
    - [Run](#image_dashboard_run)
        - [Interactive dashboard](#image_dashboard_run_dashboard)
        - [Bash](#image_dashboard_run_bash)

**NB:** You may need to run commands below with `sudo`, unless you
[manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user).

<a name="image_dashboard"></a>
## Dashboard image

<a name="image_dashboard_download"></a>
### Download
```
$ docker pull ctuning/image-classification-tflite.dashboard.ubuntu-18.04
```

<a name="image_dashboard_build"></a>
### Build
```bash
$ ck build docker:image-classification-tflite.dashboard.ubuntu-18.04
```
**NB:** Equivalent to:
```bash
$ cd `ck find docker:image-classification-tflite.dashboard.ubuntu-18.04`
$ docker build -f Dockerfile -t ctuning/image-classification-tflite.dashboard.ubuntu-18.04 .
```


<a name="image_dashboard_run"></a>
### Run

<a name="image_dashboard_run_dashboard"></a>
#### Dashboard
1. Run a dashboard container with an interactive shell:
```
$ ck run docker:image-classification-tflite.dashboard.ubuntu-18.04
```
**NB:** Equivalent to:
```bash
$ docker run -it --publish 3355:3344 --rm ctuning/image-classification-tflite.dashboard.ubuntu-18.04
```

2. Point your browser to http://localhost:3355/?template=dashboard&scenario=mlperf.mobilenets to
listen to the server.

<a name="image_dashboard_run_bash"></a>
#### Bash
```bash
$ docker run -it --rm ctuning/image-classification-tflite.dashboard.ubuntu-18.04 bash
```
