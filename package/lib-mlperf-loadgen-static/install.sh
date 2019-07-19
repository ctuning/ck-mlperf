#! /bin/bash

#
# CK installation script
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

# PACKAGE_DIR
# INSTALL_DIR

export SOURCE_DIR=${CK_ENV_MLPERF_INFERENCE_LOADGEN}
export BUILD_DIR=${INSTALL_DIR}/build
export INCLUDE_DIR=${INSTALL_DIR}/include
export LIB_DIR=${INSTALL_DIR}/lib

echo "LoadGen source directory: ${SOURCE_DIR}"
echo "LoadGen build directory: ${BUILD_DIR}"

#rm -rf ${BUILD_DIR} ${INCLUDE_DIR} ${LIB_DIR}
mkdir -p ${BUILD_DIR} ${INCLUDE_DIR} ${LIB_DIR}

cp -f ${SOURCE_DIR}/loadgen.h ${INCLUDE_DIR}
cp -f ${SOURCE_DIR}/query_sample.h ${INCLUDE_DIR}
cp -f ${SOURCE_DIR}/query_sample_library.h ${INCLUDE_DIR}
cp -f ${SOURCE_DIR}/system_under_test.h ${INCLUDE_DIR}
cp -f ${SOURCE_DIR}/test_settings.h ${INCLUDE_DIR}

cd ${BUILD_DIR}
echo ${PWD}
cmake ${SOURCE_DIR}
cmake --build ${BUILD_DIR}
cp -f ${BUILD_DIR}/libmlperf_loadgen.a ${LIB_DIR}
