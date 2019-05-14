FROM ubuntu:18.04

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

## Install all packages and immediately clean up to make the image smaller
#
RUN apt-get update -y \
    && apt-get install -y python3 python3-pip git zip bzip2 sudo wget vim curl \
    libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev python-pillow build-essential

# Install the core Collective Knowledge (CK) module.
ENV CK_ROOT=/ck-master \
    CK_REPOS=/CK \
    CK_TOOLS=/CK-TOOLS \
    PATH=${CK_ROOT}/bin:${PATH} \
    CK_PYTHON=python3 \
    LANG=C.UTF-8 \
    CK_PLATFORM_NAME="generic-linux " \
    CPP_COMPILER=gcc

RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}Ы
RUN git config --global user.name "Dividiti" && git config --global user.email "info@dividiti.com"
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT} && ${CK_PYTHON} setup.py install && ${CK_PYTHON} -c "import ck.kernel as ck"

# Install other CK modules.
RUN ck pull repo:ck-tensorflow
RUN ck pull repo:ck-mlperf

RUN echo "$CK_PLATFORM_NAME" | ck detect platform.os --update_platform_init
# Dependencies of this particular Hackathon.
RUN rm -f `which pip` && ${CK_PYTHON} -m pip install --ignore-installed pip setuptools
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`
RUN ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,scipy
RUN ck install package --tags=lib,python-package,pillow
RUN ck detect soft:compiler.gcc --full_path=`which ${CPP_COMPILER}`
RUN ck install package --tags=tool,cmake,v3.14.3

# Maybe flatbuffers is unneeded
RUN ck install package --tags=lib,flatbuffers
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1
RUN ck install package --tags=lib,xopenme

RUN ck install package:imagenet-2012-val-min
RUN ck install package:imagenet-2012-aux
RUN ck install package:model-tf-mlperf-mobilenet

RUN ck compile program:image-classification-tflite

ENTRYPOINT ["/bin/bash"]
CMD ["ck run program:image-classification-tflite"]
