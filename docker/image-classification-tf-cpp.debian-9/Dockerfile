FROM debian:9

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# TF-C++ needs: autoconf, autogen, libtool.
RUN apt update -y \
 && apt install -y apt-utils \
 && apt upgrade -y \
 && apt install -y \
        git wget zip libz-dev curl vim \
        autoconf autogen libtool \
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

# Install C/C++ dependencies.
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
RUN ck install package --tags=lib,tensorflow,vstatic,v1.13.1 --no_tags=lite

# Download and preprocess the first 500 images of the ImageNet 2012 validation dataset.
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,val,preprocessed

# Download the MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-cpp#install-the-mobilenet-model-for-tensorflow-c
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,non-quantized,from-zenodo
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,quantized,from-google
# Download the ResNet TF model.
# https://github.com/mlperf/inference/tree/master/edge/object_classification/mobilenets/tf-cpp#install-the-resnet-model
RUN ck install package --tags=image-classification,model,tf,mlperf,resnet,downloaded,from-zenodo

# Compile the Image Classification TF-C++ program.
RUN ck compile program:image-classification-tf-cpp

# Run the Image Classification TF-C++ program once with the non-quantized MobileNet model.
CMD ["ck run program:image-classification-tf-cpp --dep_add_tags.weights=mobilenet,non-quantized --env.CK_BATCH_COUNT=2"]
