#!/bin/bash

INSTALL_SCRIPT_NAME=install.sh

echo "${INSTALL_SCRIPT_NAME} : Copying flatlabels.txt to ${INSTALL_DIR}..."
cp ${ORIGINAL_PACKAGE_DIR}/flatlabels.txt ${INSTALL_DIR}

echo "${INSTALL_SCRIPT_NAME} : Done."
