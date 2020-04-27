FROM ubuntu:18.04

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# OpenVINO needs: CMake.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
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
    CK_PYTHON=python3.6 \
    GIT_USER="dividiti" \
    GIT_EMAIL="info@dividiti.com" \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name ${GIT_USER} && git config --global user.email ${GIT_EMAIL}
RUN git clone https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT}\
 && ${CK_PYTHON} setup.py install --user\
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"\
 && chmod -R g+rx /home/dvdt/.local

# Explicitly create a CK experiment entry, a folder that will be mounted
# (with '--volume=<folder_for_results>:/home/dvdt/CK_REPOS/local/experiment').
# as a shared volume between the host and the container, and make it group-writable.
# For consistency, use the "canonical" uid from ck-analytics:module:experiment.
RUN ck create_entry --data_uoa=experiment --data_uid=bc0409fb61f0aa82 --path=${CK_REPOS}/local\
 && chmod -R g+w ${CK_REPOS}/local/experiment

# Pull CK repositories (including ck-mlperf, ck-env, ck-autotuning, ck-tensorflow, ck-docker).
RUN ck pull repo:ck-openvino

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`

# Detect CMake build tool.
RUN ck detect soft --tags=cmake --full_path=`which cmake`

# Detect Python.
RUN ck detect soft --tags=compiler,python --full_path=`which ${CK_PYTHON}`
# Install the latest Python package installer (pip) and some dependencies.
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user


#-----------------------------------------------------------------------------#
# Step 1. Install Python dependencies (for Model Optimizer and LoadGen).
#-----------------------------------------------------------------------------#
# OpenVINO pre-release strictly requires TensorFlow < 2.0 and NetworkX < 2.4.
RUN ck install package --tags=lib,python-package,tensorflow --force_version=1.15.2
RUN ck install package --tags=lib,python-package,networkx --force_version=2.3.0
RUN ck install package --tags=lib,python-package,defusedxml
# Cython is an implicit dependency of NumPy.
RUN ck install package --tags=lib,python-package,cython
RUN ck install package --tags=lib,python-package,numpy
# test-generator is an implicit dependency of Model Optimizer (not in requirements.txt).
RUN ck install package --tags=lib,python-package,test-generator
# Abseil is a LoadGen dependency.
RUN ck install package --tags=lib,python-package,absl


#-----------------------------------------------------------------------------#
# Step 2. Install C++ dependencies (for Inference Engine and MLPerf program).
#-----------------------------------------------------------------------------#
RUN ck install package --tags=channel-stable,opencv,v3.4.3
RUN ck install package --tags=channel-stable,boost,v1.67.0 --no_tags=min-for-caffe
# Install LoadGen from a branch reconstructed according to Intel's README.
RUN ck install package --tags=mlperf,inference,source,dividiti.v0.5-intel
RUN ck install package --tags=lib,loadgen,static
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 3. Install the OpenVINO "pre-release" used for MLPerf Inference v0.5.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=lib,openvino,pre-release
RUN ck compile ck-openvino:program:mlperf-inference-v0.5
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 4. Install the first 500 images of the ImageNet 2012 validation dataset.
# TODO: Create a calibration dataset.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=dataset,imagenet,val,min --no_tags=resized
RUN ck install package --tags=dataset,imagenet,aux
# The OpenVINO program expects to find val_map.txt in the dataset directory.
RUN head -n 500 `ck locate env --tags=aux`/val.txt > `ck locate env --tags=val`/val_map.txt
# Install misc Python dependencies required for calibration.
RUN ${CK_PYTHON}  -m pip install --user \
    nibabel pillow progress py-cpuinfo pyyaml shapely sklearn tqdm xmltodict yamlloader
# Install "headless" OpenCV (which doesn't need libsm6, libxext6, libxrender-dev).
RUN ck install package --tags=lib,python-package,cv2,opencv-python-headless
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 5. Install the official ResNet model for MLPerf Inference v0.5
# and convert it into the OpenVINO format.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=image-classification,model,tf,mlperf,resnet
RUN ck install package --tags=model,openvino,resnet50
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 6. Install the official MobileNet model for MLPerf Inference v0.5
# and convert it into the OpenVINO format.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=image-classification,model,tf,mobilenet-v1-1.0-224,non-quantized
RUN ck install package --tags=model,openvino,mobilenet
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 7. INTENTIONALLY LEFT BLANK.
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 8. INTENTIONALLY LEFT BLANK.
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 9. Make final preparations to run the OpenVINO program.
#-----------------------------------------------------------------------------#
# Allow the program create tmp files when running under an external user.
RUN chmod -R g+rwx `ck find ck-openvino:program:mlperf-inference-v0.5`
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Run the OpenVINO program that Intel prepared for MLPerf Inference v0.5
# with the quantized ResNet model
# on the first 500 images of the ImageNet 2012 validation dataset
# using all (virtual) CPU cores.
#-----------------------------------------------------------------------------#
CMD ["export NPROCS=`grep -c processor /proc/cpuinfo` \
 && ck run ck-openvino:program:mlperf-inference-v0.5 --skip_print_timers \
    --cmd_key=image-classification --env.CK_OPENVINO_MODEL_NAME=resnet50 \
    --env.CK_LOADGEN_SCENARIO=Offline --env.CK_LOADGEN_MODE=Accuracy --env.CK_LOADGEN_DATASET_SIZE=500 \
    --env.CK_OPENVINO_NTHREADS=$NPROCS --env.CK_OPENVINO_NSTREAMS=$NPROCS --env.CK_OPENVINO_NIREQ=$NPROCS \
 && cat /home/dvdt/CK_REPOS/ck-openvino/program/mlperf-inference-v0.5/tmp/accuracy.txt"]
