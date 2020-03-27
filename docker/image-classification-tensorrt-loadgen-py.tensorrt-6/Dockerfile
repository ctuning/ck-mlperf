# 19.12-py3 is the last image to come with TensorRT 6.
# 20.01-py3 comes with TensorRT 7.
FROM nvcr.io/nvidia/tensorrt:19.12-py3

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# TF Object Detection API needs ProtoBuf 3.0 which needs CMake.
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

# Create a non-root user with a fixed group id 1500 and a fixed user id 2000
# (hopefully distinct from any host user id for security reasons).
# See the README for the gory details.
RUN groupadd -g 1500 dvdtg
RUN useradd -u 2000 -g dvdtg --create-home --shell /bin/bash dvdt
USER dvdt:dvdtg
WORKDIR /home/dvdt

# Install Collective Knowledge (CK). Make it group-executable.
ENV CK_ROOT=/home/dvdt/CK \
    CK_REPOS=/home/dvdt/CK_REPOS \
    CK_TOOLS=/home/dvdt/CK_TOOLS \
    PATH=${CK_ROOT}/bin:/home/dvdt/.local/bin:${PATH} \
    CK_CC=gcc \
    CK_PYTHON=python3 \
    GIT_USER="dividiti" \
    GIT_EMAIL="info@dividiti.com" \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name ${GIT_USER} && git config --global user.email ${GIT_EMAIL}
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT}\
 && ${CK_PYTHON} setup.py install --user\
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"\
 && chmod -R g+rx /home/dvdt/.local \
 && umask 002 && chgrp dvdtg ${CK_TOOLS} && chmod g+ws ${CK_TOOLS}

# Explicitly create a CK experiment entry, a folder that will be mounted
# (with '--volume=<folder_for_results>:/home/dvdt/CK_REPOS/local/experiment').
# as a shared volume between the host and the container, and make it group-writable.
# For consistency, use the "canonical" uid from ck-analytics:module:experiment.
RUN ck create_entry --data_uoa=experiment --data_uid=bc0409fb61f0aa82 --path=${CK_REPOS}/local\
 && chmod -R g+w ${CK_REPOS}/local/experiment

# Pull CK repositories (including ck-mlperf, ck-env, ck-autotuning, ck-tensorflow, ck-docker).
RUN ck pull repo:ck-mlperf
RUN ck pull repo:ck-tensorrt

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

#-----------------------------------------------------------------------------#
# Step 1. Install Python packages.
#-----------------------------------------------------------------------------#
# Detect Python.
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`
# Install the latest Python package installer (pip) and some dependencies.
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user
RUN ck install package --tags=python-package,cython
RUN ck install package --tags=python-package,numpy
RUN ck install package --tags=python-package,absl
RUN ck install package --tags=python-package,opencv-python-headless

#-----------------------------------------------------------------------------#
# Step 2. Detect GCC, CUDA and install PyCUDA.
#-----------------------------------------------------------------------------#
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
RUN ck detect soft:compiler.cuda --full_path=/usr/local/cuda/bin/nvcc
RUN ck install package --tags=python-package,pycuda

#-----------------------------------------------------------------------------#
# Step 3. Detect TensorRT and PyTensorRT.
#-----------------------------------------------------------------------------#
RUN ck detect soft:lib.tensorrt --full_path=/usr/lib/x86_64-linux-gnu/libnvinfer.so
RUN ck detect soft:lib.python.tensorrt --full_path=/usr/lib/python3.6/dist-packages/tensorrt/__init__.py

#-----------------------------------------------------------------------------#
# Step 4. Download and preprocess the first 500 images of the ImageNet 2012
# validation dataset.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,val,full,preprocessed,using-opencv

#-----------------------------------------------------------------------------#
# Step 5. Install MLPerf Inference packages.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=mlperf,inference,source
RUN ck install package --tags=mlperf,loadgen,python-package

#-----------------------------------------------------------------------------#
# Step 6. Make final preparations to run Image Classification with TensorRT.
#-----------------------------------------------------------------------------#
# Install the official MLPerf ResNet ONNX model.
RUN ck install package --tags=model,onnx,resnet,downloaded
# Allow to create tmp files when running under an external user.
RUN chmod -R g+rwx `ck find program:image-classification-tensorrt-loadgen-py`
# Allow to create new environments when running under an external user.
RUN chmod -R g+rwx $CK_REPOS/local/env

## Print CUDA device info.
#RUN ck compile program:tool-print-cuda-devices
#CMD [ "ck run program:tool-print-cuda-devices" ]

RUN ck pull repo:ck-tensorrt

CMD [ "ck run program:image-classification-tensorrt-loadgen-py --skip_print_timers --env.CK_SILENT_MODE \
--env.CK_LOADGEN_CONF_FILE=`ck find program:image-classification-tensorrt-loadgen-py`/user.conf \
--env.CK_LOADGEN_MODE=AccuracyOnly --env.CK_LOADGEN_SCENARIO=MultiStream \
--env.CK_LOADGEN_DATASET_SIZE=500 --env.CK_LOADGEN_BUFFER_SIZE=500 \
--env.CK_LOADGEN_MULTISTREAMNESS=32 --env.CK_BATCH_SIZE=32 \
--dep_add_tags.weights=model,resnet,converted-from-onnx,fp32,maxbatch.32 \
&& head -n 12 `ck find program:image-classification-tensorrt-loadgen-py`/tmp/mlperf_log_summary.txt" ]
