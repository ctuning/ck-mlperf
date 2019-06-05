FROM ubuntu:18.04

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively. FIXME: Interferes with the dashboard?
#ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# Running the dashboard in the background needs: daemonize.
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
 curl\
 vim\
 daemonize\
 python3 python3-pip\
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

# Create a repository for benchmarking results.
RUN ck create repo:mlperf-mobilenets --quiet

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect Python.
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`

# Install the latest Python package installer (pip).
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --upgrade --user
# Install Python dependencies.
RUN ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,scipy --force_version=1.2.1
RUN ck install package --tags=lib,python-package,pillow
# Install pandas for dashboard.
RUN ${CK_PYTHON} -m pip install pandas --user

# Install TFLite.
RUN ck install package --tags=lib,tensorflow-lite,tensorflow-static,v1.13.1

# Download and preprocess the first 500 images of the ImageNet 2012 validation dataset.
RUN ck install package --tags=dataset,imagenet,aux
RUN ck install package --tags=dataset,imagenet,val,original,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,val,preprocessed

# Download the MobileNet TF/TFLite models (non-quantized and quantized).
# https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#install-the-mobilenet-models-for-tflite
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,non-quantized,from-zenodo
RUN ck install package --tags=image-classification,model,tf,tflite,mlperf,mobilenet,quantized,from-google

# Benchmark the performance of the non-quantized MobileNet model.
RUN ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--dep_add_tags.weights=mlperf,image-classification,mobilenet,non-quantized,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-mobilenet-non-quantized-tflite-performance \
--tags=mlperf,image-classification,mobilenet,non-quantized,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
# Benchmark the accuracy of the non-quantized MobileNet model.
RUN ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--dep_add_tags.weights=mlperf,image-classification,mobilenet,non-quantized,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-mobilenet-non-quantized-tflite-accuracy \
--tags=mlperf,image-classification,mobilenet,non-quantized,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys

# Benchmark the performance of the quantized MobileNet model.
RUN ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--dep_add_tags.weights=mlperf,image-classification,mobilenet,quantized,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-mobilenet-quantized-tflite-performance \
--tags=mlperf,image-classification,mobilenet,quantized,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
# Benchmark the accuracy of the quantized MobileNet model.
RUN ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--dep_add_tags.weights=mlperf,image-classification,mobilenet,quantized,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-mobilenet-quantized-tflite-accuracy \
--tags=mlperf,image-classification,mobilenet,quantized,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys

# Download the ResNet TFLite models (with and without the ArgMax operator).
# https://github.com/mlperf/inference/blob/master/edge/object_classification/mobilenets/tflite/README.md#install-the-resnet-model
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,with-argmax
RUN ck install package --tags=image-classification,model,tflite,mlperf,resnet,downloaded,no-argmax

# Benchmark the performance of the ResNet model with the ArgMax operator.
RUN ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--dep_add_tags.weights=mlperf,image-classification,resnet,with-argmax,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-resnet-with-argmax-tflite-performance \
--tags=mlperf,image-classification,resnet,with-argmax,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
# Benchmark the accuracy of the ResNet model with the ArgMax operator.
RUN ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--dep_add_tags.weights=mlperf,image-classification,resnet,with-argmax,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-resnet-with-argmax-tflite-accuracy \
--tags=mlperf,image-classification,resnet,with-argmax,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys

# Benchmark the performance of the ResNet model without the ArgMax operator.
RUN ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--dep_add_tags.weights=mlperf,image-classification,resnet,no-argmax,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-resnet-no-argmax-tflite-performance \
--tags=mlperf,image-classification,resnet,no-argmax,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
# Benchmark the accuracy of the ResNet model without the ArgMax operator.
RUN ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=500 \
--dep_add_tags.weights=mlperf,image-classification,resnet,no-argmax,tflite \
--record --record_repo=mlperf-mobilenets --record_uoa=mlperf-image-classification-resnet-no-argmax-tflite-accuracy \
--tags=mlperf,image-classification,resnet,no-argmax,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys

# This command spawns the server in the background (daemon) mode, while
# also brings up an interactive shell in the same container.
CMD echo -e "Point your browser to: http://localhost:3355/?template=dashboard&scenario=mlperf.mobilenets"\
 && daemonize -o ${HOME}/ck_server.out -e ${HOME}/ck_server.err\
 `which ck` display dashboard --scenario=mlperf.mobilenets\
 --host=0.0.0.0 --wfe_host=localhost --wfe_port=3355\
 && /bin/bash
