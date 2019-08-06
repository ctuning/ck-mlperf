FROM debian:9

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# Install to share with other images: cmake.
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
 curl\
 cmake\
 python3 python3-pip\
 vim\
 && apt clean

# Create non-root user.
RUN useradd --create-home --user-group --shell /bin/bash dvdt
USER dvdt:dvdt
WORKDIR /home/dvdt

# Install Collective Knowledge (CK).
ENV CK_ROOT=/home/dvdt/CK \
    CK_REPOS=/home/dvdt/CK_REPOS \
    CK_TOOLS=/home/dvdt/CK_TOOLS \
    PATH=${CK_ROOT}/bin:/home/dvdt/.local/bin:${PATH} \
    CK_PYTHON=python3 \
    CK_CC=gcc \
    GIT_USER="dividiti" \
    GIT_EMAIL="info@dividiti.com" \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name ${GIT_USER} && git config --global user.email ${GIT_EMAIL}
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT} \
 && ${CK_PYTHON} setup.py install --user \
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"

# Pull CK repositories (including ck-env, ck-autotuning and ck-tensorflow).
RUN ck pull repo:ck-mlperf

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
# Install TFLite.
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1

# Detect Python.
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`
# Install the latest Python package installer (pip).
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user
# Install Python dependencies.
RUN ck install package --tags=lib,python-package,numpy

# Download the MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite/README.md#mobilenet-non-quantized
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,non-quantized,from-zenodo
# https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite/README.md#mobilenet-quantized
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,quantized,from-zenodo

# Download the ResNet TFLite models (with and without the ArgMax operator).
# https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite/README.md#resnet
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,with-argmax
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,no-argmax

# Download and preprocess the first 500 images of the ImageNet 2012 validation dataset.
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
# Preprocess using "headless" OpenCV (which doesn't need libsm6, libxext6, libxrender-dev).
RUN ck install package --tags=lib,python-package,cv2,opencv-python-headless
RUN ck install package --tags=dataset,imagenet,val,preprocessed,using-opencv

# Compile the Image Classification TFLite program.
RUN ck compile program:image-classification-tflite

# Run the Image Classification TFLite program
# with the non-quantized MobileNet model twice.
CMD ["ck run program:image-classification-tflite \
--dep_add_tags.images=preprocessed,using-opencv \
--dep_add_tags.weights=mobilenet,non-quantized \
--env.CK_BATCH_COUNT=2"]
