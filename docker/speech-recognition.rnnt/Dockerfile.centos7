FROM centos:7

LABEL maintainer="Anton Lokhmotov <anton@dividiti.com>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip, unzip.
# Python 3 needs: open-ssl-devel, bzip2-devel, libffi-devel.
RUN yum upgrade -y\
 && yum install -y\
 make which patch gcc gcc-c++\
 git wget zip unzip\
 openssl-devel bzip2-devel libffi-devel\
 && yum clean all

# Install Python 3.
RUN cd /usr/src \
 && wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz \
 && tar xzf Python-3.7.7.tgz \
 && rm Python-3.7.7.tgz \
 && cd Python-3.7.7 \
 && ./configure --enable-optimizations \
 && make altinstall \
 && cd /usr/src \
 && rm -rf Python-3.7.7*

# Create non-root user.
RUN useradd --create-home --user-group --shell /bin/bash dvdt
USER dvdt:dvdt
WORKDIR /home/dvdt

# Install Collective Knowledge (CK).
ENV CK_ROOT=/home/dvdt/CK \
    CK_REPOS=/home/dvdt/CK_REPOS \
    CK_TOOLS=/home/dvdt/CK_TOOLS \
    PATH=${CK_ROOT}/bin:/home/dvdt/.local/bin:${PATH} \
    CK_PYTHON=python3.7 \
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
RUN ck pull repo:ck-pytorch

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=`which ${CK_CC}`
# Detect Python.
RUN ck detect soft:compiler.python --full_path=`which ${CK_PYTHON}`
# Install the latest Python package installer (pip).
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user

# Run the Speech Recognition RNNT program.
CMD ["echo TODO."]
