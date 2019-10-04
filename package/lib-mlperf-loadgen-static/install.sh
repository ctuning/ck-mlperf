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
read -d '' CMK_CMD <<EO_CMK_CMD
${CK_ENV_TOOL_CMAKE_BIN}/cmake "${SOURCE_DIR}" \
  -DCMAKE_C_COMPILER="${CK_CC_PATH_FOR_CMAKE}" \
  -DCMAKE_C_FLAGS="${CK_CC_FLAGS_FOR_CMAKE} ${EXTRA_FLAGS}" \
  -DCMAKE_CXX_COMPILER="${CK_CXX_PATH_FOR_CMAKE}" \
  -DCMAKE_CXX_FLAGS="${CK_CXX_FLAGS_FOR_CMAKE} ${CK_COMPILER_FLAG_CPP14} ${EXTRA_FLAGS} -Wno-psabi" \
  -DCMAKE_AR="${CK_AR_PATH_FOR_CMAKE}" \
  -DCMAKE_RANLIB="${CK_RANLIB_PATH_FOR_CMAKE}" \
  -DCMAKE_LINKER="${CK_LD_PATH_FOR_CMAKE}" \
  -DPYTHON_EXECUTABLE="${CK_ENV_COMPILER_PYTHON_FILE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
EO_CMK_CMD

# Print the EXACT command we are about to run (with all interpolations that have not been specifically blocked)
echo $CMK_CMD
echo ""

# Now run it from the build directory.
cd ${BUILD_DIR}
eval $CMK_CMD
exit_if_error

# Build and install.
cmake \
  --build ${BUILD_DIR} \
  --target install
exit_if_error

# Clean up.
rm -rf ${BUILD_DIR}
exit_if_error
