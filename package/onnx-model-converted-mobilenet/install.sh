#!/bin/bash

$CK_ENV_COMPILER_PYTHON_FILE -m tf2onnx.convert --input $CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH --inputs $MODEL_INPUT_LAYER_NAME --inputs-as-nchw $MODEL_INPUT_LAYER_NAME --outputs $MODEL_OUTPUT_LAYER_NAME --fold_const --verbose --opset 7 --output ${INSTALL_DIR}/${PACKAGE_NAME}
