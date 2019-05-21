## [Ubuntu](https://hub.docker.com/_/ubuntu/) 18.04

**NB:** `#` means execution under root or with `sudo`.

### Change directory
```
$ cd `ck find docker:image-classification-tf-py-ubuntu-18.04`
```

### Prepare
Due to NVIDIA restrictions, You should register (free) at https://developer.nvidia.com/
for [downloading](https://developer.nvidia.com/rdp/cudnn-download) required `cuDNN` packages.

You should download cuDNN v7.5.1 for CUDA 10.0:
 - [cuDNN Runtime Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7_7.5.1.10-1%2Bcuda10.0_amd64.deb)
 - [cuDNN Developer Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.0_20190418/Ubuntu18_04-x64/libcudnn7-dev_7.5.1.10-1%2Bcuda10.0_amd64.deb)

 After that You should create `cuda.tar` with following structure:
```
cuda.tar:
 /cuda
   /lib
     Ubuntu18_04-x64/libcudnn7_7.5.1.10-1+cuda10.0_amd64.deb
   /lib-dev
     Ubuntu18_04-x64/libcudnn7-dev_7.5.1.10-1+cuda10.0_amd64.deb
```
and put `cuda.tar` in current directory

### Build image
```
# docker build . -f Dockerfile.cuda -t image-classification-tf-py-ubuntu-18.04
```

### Check image 
#### View layer-by-layer build history
```
# docker history image-classification-tf-py-ubuntu-18.04
```
#### View space usage
```
# docker system df -v
```

### Run image

#### Image Classification (default)
```
# docker run --rm image-classification-tf-py-ubuntu-18.04
```
**NB:** Equivalent to:
```
# docker run --rm image-classification-tf-py-ubuntu-18.04 \
"ck run program:image-classification-tf-py"
```

#### Image Classification (custom)
```
# docker run --rm image-classification-tf-py-ubuntu-18.04 \
"ck run program:image-classification-tf-py --env.CK_BATCH_COUNT=10"
```

#### Bash
```
# docker run -it --rm image-classification-tf-py-ubuntu-18.04 bash
```
