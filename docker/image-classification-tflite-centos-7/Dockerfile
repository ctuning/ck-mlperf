FROM centos:centos7

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

SHELL ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, zip, wget, unzip.
# TF needs: curl.
# Python 3: open-ssl-devel, bzip2-devel, libffi-devel 
RUN yum update -y \
 && yum install -y \
       sudo git vim \
       gcc make patch gcc-c++ \
       openssl-devel bzip2-devel libffi-devel\
       wget which zip unzip \
 && yum clean all

# Install Python 3
RUN cd /usr/src \
 && wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz \
 && tar xzf Python-3.7.3.tgz \
 && rm Python-3.7.3.tgz \
 && cd Python-3.7.3 \
 && ./configure --enable-optimizations \
 && make altinstall \
 && cd /usr/src \
 && rm -rf Python-3.7.3*

# Create non-root user
RUN useradd --create-home --user-group  --shell /bin/bash dvdt
USER dvdt:dvdt
WORKDIR /home/dvdt

# Install Collective Knowledge (CK).
ENV CK_ROOT=/home/dvdt/CK \
    CK_REPOS=/home/dvdt/CK_REPOS \
    CK_TOOLS=/home/dvdt/CK_TOOLS \
    PATH=${CK_ROOT}/bin:/home/dvdt/.local/bin:${PATH} \
    CK_PYTHON=python3.7 \
    CK_CC=gcc \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name "dividiti" && git config --global user.email "info@dividiti.com"
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT} \
 && ${CK_PYTHON} setup.py install --user \
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"

# Pull CK repositories (including ck-env, ck-autotuning and ck-tensorflow).
RUN ck pull repo:ck-mlperf

# Use generic Linux settings.
RUN ck detect platform.os --update_platform_init --platform_init_uoa=generic-linux-dummy

# Install Python dependencies.
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`

RUN ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,scipy
RUN ck install package --tags=lib,python-package,pillow

# Install C/C++ dependencies.

RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1

# Download and preprocess the first 500 images of the ImageNet 2012 validation dataset.
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,val,preprocessed

# Download MobileNet TF/TFLite model.
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,non-quantized

# Compile Image Classification TFLite program.
RUN ck compile program:image-classification-tflite

# Run Image Classification TFLite program once.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["ck run program:image-classification-tflite"]
