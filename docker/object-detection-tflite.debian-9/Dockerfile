FROM debian:9

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# TensorFlow Object Detection API needs ProtoBuf 3.0 which needs CMake.
RUN apt update -y \
 && apt install -y apt-utils \
 && apt upgrade -y \
 && apt install -y \
        git wget zip libz-dev curl vim \
        cmake \
        python3 python3-pip \
 && apt clean

# Upgrade pip.
RUN python3 -m pip install --upgrade pip

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

# Install Python dependencies.
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`

RUN ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,scipy --force_version=1.2.1
RUN ck install package --tags=lib,python-package,pillow
RUN ck install package --tags=lib,python-package,matplotlib
RUN ck install package --tags=lib,python-package,cython

RUN ck show env --tags=python-package

# Install C/C++ dependencies.
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1

# Download the SSD-MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/blob/master/edge/object_detection/ssd_mobilenet/tflite/README.md#install-the-ssd-mobilenet-models-for-tflite
# NB: Pulls TensorFlow Object Detection API. TODO: Make this explicit?
RUN ck install package --tags=model,tflite,object-detection,mlperf,ssd-mobilenet,non-quantized

# Download the COCO 2017 validation dataset (5,000 images). Accept the default path.
RUN echo | ck install package --tags=object-detection,dataset,coco.2017,val,original,full
# Preprocess the first 50 images.
RUN ck install package --tags=object-detection,dataset,coco.2017,preprocessed,first.50
# Preprocess all the 5,000 images.
RUN ck install package --tags=object-detection,dataset,coco.2017,preprocessed,full

# Install the Python COCO API.
RUN ck install package --tags=tool,coco,api

# Compile the Object Detection TFLite program.
RUN ck compile program:object-detection-tflite

# Run the Object Detection TFLite program
# with the non-quantized SSD-MobileNet model with regular non-max-suppression
# on the first 50 images of the COCO 2017 validation dataset.
CMD ["ck run program:object-detection-tflite \
--dep_add_tags.weights=ssd-mobilenet,non-quantized --env.USE_NMS=regular \
--dep_add_tags.dataset=coco.2017,first.50 --env.CK_BATCH_COUNT=50"]
