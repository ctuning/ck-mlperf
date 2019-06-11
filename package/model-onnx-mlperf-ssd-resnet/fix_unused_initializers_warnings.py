# Code adapted from https://github.com/Microsoft/onnxruntime/issues/915
#
# execution example in virtual env:
#   ck virtual env --tags=onnx,python-package --shell_cmd="python3 fix_unused_initializers_warnings.py resnet34.onnx"

import sys
import onnx
from onnx import optimizer

onnx_filename = sys.argv[1]
onnx_model = onnx.load(onnx_filename)
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = optimizer.optimize(onnx_model, passes)
onnx.save(optimized_model, onnx_filename)
