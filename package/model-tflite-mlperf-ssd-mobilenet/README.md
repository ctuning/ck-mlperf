## MLPerf Inference - Object Detection - SSD-MobileNet (TFLite)

**TODO:** [Move model files to Zenodo](https://github.com/ctuning/ck-mlperf/issues/9).


This model was converted to TFLite from the [original](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) TF model in two steps,
by adapting instructions from [Google's blog](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193).

```
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ tar xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ ls -la ssd_mobilenet_v1_coco_2018_01_28/*
-rw-r--r-- 1 anton dvdt       77 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/checkpoint
-rw-r--r-- 1 anton dvdt 29103956 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
-rw-r--r-- 1 anton dvdt 27380740 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.data-00000-of-00001
-rw-r--r-- 1 anton dvdt     8937 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.index
-rw-r--r-- 1 anton dvdt  3006546 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/model.ckpt.meta
-rw-r--r-- 1 anton dvdt     4138 Feb  1  2018 ssd_mobilenet_v1_coco_2018_01_28/pipeline.config

ssd_mobilenet_v1_coco_2018_01_28/saved_model:
total 29020
drwxr-xr-x 3 anton dvdt     4096 Feb  1  2018 .
drwxr-xr-x 3 anton dvdt     4096 Feb  1  2018 ..
-rw-r--r-- 1 anton dvdt 29700424 Feb  1  2018 saved_model.pb
drwxr-xr-x 2 anton dvdt     4096 Feb  1  2018 variables
```
As explained below, we used both TensorFlow v1.13 and v1.11 built from source as follows:
```
$ ck pull repo:ck-tensorflow
$ ck install package:lib-tensorflow-1.13.1-src-cpu
$ ck install package:lib-tensorflow-1.11.0-src-cpu
```
**TODO:** We should try with TensorFlow v1.12 (now that we have a new package):
```
$ ck install package:lib-tensorflow-1.12.2-src-cpu
```

1. [Creating TFLite graph from TF checkpoint](#step_1)
2. [Creating TFLite model from TFLite graph](#step_2)
    1. [with the postprocessing layer](#step_2_option_1)
    2. [without the postprocessing layer](#step_2_option_2)

<a name="step_1"></a>
### Step 1: `model.ckpt.*` to `tflite_graph.pb`

In this step, we used the [TensorFlow Model API](https://github.com/tensorflow/models) and TensorFlow v1.11:

```bash
$ python object_detection/export_tflite_ssd_graph.py \
    --input_type image_tensor \
    --pipeline_config_path <TF_MODEL_DIR>/pipeline.config \
    --trained_checkpoint_prefix <TF_MODEL_DIR>/model.ckpt \
    --output_directory <TFLITE_MODEL_DIR> \
    --add_postprocessing_op=true \
    --config_override " \
            model { \
            ssd { \
              post_processing { \
                batch_non_max_suppression { \
                  score_threshold: 0.3 \
                  iou_threshold: 0.6 \
                  max_detections_per_class: 100 \
                  max_total_detections: 100 \
                } \
             } \
          } \
       } \
       "
```
**NB:** With TensorFlow v1.13, we observed the following error:

```bash
Traceback (most recent call last):
  File "object_detection/export_tflite_ssd_graph.py", line 96, in <module>
    from object_detection import export_tflite_ssd_graph_lib
  File "/home/ivan/CK-TOOLS/tensorflowmodel-api-master/models/research/object_detection/export_tflite_ssd_graph_lib.py", line 27, in <module>
    from object_detection import exporter
  File "/home/ivan/CK-TOOLS/tensorflowmodel-api-master/models/research/object_detection/exporter.py", line 20, in <module>
    from tensorflow.contrib.quantize.python import graph_matcher
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/__init__.py", line 40, in <module>
    from tensorflow.contrib import distribute
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/distribute/__init__.py", line 33, in <module>
    from tensorflow.contrib.distribute.python.tpu_strategy import TPUStrategy
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/distribute/python/tpu_strategy.py", line 27, in <module>
    from tensorflow.contrib.tpu.python.ops import tpu_ops
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/tpu/__init__.py", line 73, in <module>
    from tensorflow.contrib.tpu.python.tpu.keras_support import tpu_model as keras_to_tpu_model
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/tpu/python/tpu/keras_support.py", line 62, in <module>
    from tensorflow.contrib.tpu.python.tpu import tpu
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/tpu/python/tpu/tpu.py", line 24, in <module>
    from tensorflow.contrib.compiler import xla
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/contrib/compiler/xla.py", line 28, in <module>
    from tensorflow.python.estimator import model_fn as model_fn_lib
  File "/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.13.1-compiler.python-3.6.7-linux-64/lib/tensorflow/python/estimator/__init__.py", line 26, in <module>
    from tensorflow_estimator.python import estimator
ModuleNotFoundError: No module named 'tensorflow_estimator'
```

<a name="step_2"></a>
### Step 2: from `tflite_graph.pb` to `detect*.tflite`
In this step, we used TensorFlow v1.13.

<a name="step_2_option_1"></a>
#### Option 1: from `tflite_graph.pb` to `detect.tflite`

```bash
$ bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=<TFLITE_MODEL_DIR>/tflite_graph.pb \
    --output_file=<TFLITE_MODEL_DIR>/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=FLOAT \
    --allow_custom_ops
...
INFO: Invocation ID: b49af84b-4ef0-44c9-adf0-236700f8cd86
INFO: Analysed target //tensorflow/lite/toco:toco (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //tensorflow/lite/toco:toco up-to-date:
  bazel-bin/tensorflow/lite/toco/toco
INFO: Elapsed time: 0.247s, Critical Path: 0.00s
INFO: 0 processes.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/tensorflow/lite/toco/toco '--input_file=/home/ivan/Downloads/tflite_ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb' '--output_file=/home/ivan/Downloads/tflite_ssd_mobilenet_v1_coco_2018_01_28/_n2_detect.tflite' '--input_shapes=1,300,300,3' '--input_arrays=normalized_input_image_tensor' '--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' '--infeINFO: Build completed successfully, 1 total action
2019-04-22 09:11:44.344086: I tensorflow/lite/toco/import_tensorflow.cc:1324] Converting unsupported operation: TFLite_Detection_PostProcess
2019-04-22 09:11:44.354655: I tensorflow/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 500 operators, 754 arrays (0 quantized)
2019-04-22 09:11:44.366436: I tensorflow/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 500 operators, 754 arrays (0 quantized)
2019-04-22 09:11:44.401719: I tensorflow/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 1: 64 operators, 176 arrays (0 quantized)
2019-04-22 09:11:44.402990: I tensorflow/lite/toco/graph_transformations/graph_transformations.cc:39] Before dequantization graph transformations: 64 operators, 176 arrays (0 quantized)
2019-04-22 09:11:44.405046: I tensorflow/lite/toco/allocate_transient_arrays.cc:345] Total transient array allocated size: 11520000 bytes, theoretical optimal value: 8640000 bytes.
2019-04-22 09:11:44.405323: I tensorflow/lite/toco/toco_tooling.cc:399] Estimated count of arithmetic ops: 2.49483 billion (note that a multiply-add is counted as 2 ops).
2019-04-22 09:11:44.405706: W tensorflow/lite/toco/tflite/operator.cc:1407] Ignoring unsupported type in list attribute with key '_output_types'
```

**NB:** With TensorFlow v1.11, we observed more warnings:
```
...
WARNING: The following rc files are no longer being read, please transfer their contents or import their path into one of the standard rc files:
/home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.11.0-compiler.python-3.6.7-linux-64/src/tools/bazel.rc
Starting local Bazel server and connecting to it...
INFO: Invocation ID: d348f3c7-0031-430f-af13-a24d406c5be2
WARNING: /home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.11.0-compiler.python-3.6.7-linux-64/src/tensorflow/core/BUILD:2463:1: in includes attribute of cc_library rule //tensorflow/core:framework_internal_headers_lib: '../../external/com_google_absl' resolves to 'external/com_google_absl' not below the relative path of its package 'tensorflow/core'. This will be an error in the future. Since this rule was created by the macro 'cc_header_only_library', the error might have been caused by the macro implementation in /home/ivan/CK-TOOLS/lib-tensorflow-src-cpu-1.11.0-compiler.python-3.6.7-linux-64/src/tensorflow/tensorflow.bzl:1373:20
INFO: Analysed target //tensorflow/contrib/lite/toco:toco (58 packages loaded, 3088 targets configured).
INFO: Found 1 target...
Target //tensorflow/contrib/lite/toco:toco up-to-date:
  bazel-bin/tensorflow/contrib/lite/toco/toco
INFO: Elapsed time: 7.240s, Critical Path: 0.32s
INFO: 0 processes.
INFO: Build completed successfully, 1 total action
INFO: Running command line: bazel-bin/tensorflow/contrib/lite/toco/toco '--input_file=/home/ivan/Downloads/tflite_ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb' '--output_file=/home/ivan/Downloads/tflite_ssd_mobilenet_v1_coco_2018_01_28/_n_detect.tflite' '--input_shapes=1,300,300,3' '--input_arrays=normalized_input_image_tensor' '--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3' INFO: Build completed successfully, 1 total action
2019-04-22 08:56:09.292247: I tensorflow/contrib/lite/toco/import_tensorflow.cc:1055] Converting unsupported operation: TFLite_Detection_PostProcess
2019-04-22 08:56:09.296418: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 500 operators, 754 arrays (0 quantized)
2019-04-22 08:56:09.308100: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 500 operators, 754 arrays (0 quantized)
2019-04-22 08:56:09.345671: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 1: 64 operators, 176 arrays (0 quantized)
2019-04-22 08:56:09.346852: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before dequantization graph transformations: 64 operators, 176 arrays (0 quantized)
2019-04-22 08:56:09.347843: I tensorflow/contrib/lite/toco/allocate_transient_arrays.cc:345] Total transient array allocated size: 11520000 bytes, theoretical optimal value: 8640000 bytes.
2019-04-22 08:56:09.348102: I tensorflow/contrib/lite/toco/toco_tooling.cc:388] Estimated count of arithmetic ops: 2.49483 billion (note that a multiply-add is counted as 2 ops).
2019-04-22 08:56:09.348390: W tensorflow/contrib/lite/toco/tflite/operator.cc:1219] Ignoring unsupported type in list attribute with key '_output_types'
```

<a name="step_2_option_2"></a>
#### Option 2: from `tflite_graph.pb` to `detect_cut.tflite`

```bash
$ bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=<TFLITE_MODEL_DIR>/tflite_graph.pb \
    --output_file=<TFLITE_MODEL_DIR>/detect_cut.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='raw_outputs/box_encodings','raw_outputs/class_predictions' \
    --inference_type=FLOAT \
    --allow_custom_ops
```
