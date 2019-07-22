#! /bin/bash

#
# CK installation script
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

# PACKAGE_DIR
# INSTALL_DIR

function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}

export SOURCE_DIR=${CK_ENV_MLPERF_INFERENCE_LOADGEN}
export BUILD_DIR=${INSTALL_DIR}/build
export INCLUDE_DIR=${INSTALL_DIR}/include
export LIB_DIR=${INSTALL_DIR}/lib

echo "LoadGen source directory: ${SOURCE_DIR}"
echo "LoadGen build directory: ${BUILD_DIR}"

rm -rf ${BUILD_DIR} ${INCLUDE_DIR} ${LIB_DIR}
mkdir -p ${BUILD_DIR} ${INCLUDE_DIR} ${LIB_DIR}

# Configure.
cd ${BUILD_DIR}
cmake ${SOURCE_DIR} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
exit_if_error

# Build and install.
cmake \
  --build ${BUILD_DIR} \
  --target install
exit_if_error

# Clean up.
rm -rf ${BUILD_DIR}
exit_if_error
