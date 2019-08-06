# MLPerf Inference - Object Detection - SSD-MobileNet (TFLite)

This model was converted to TFLite from the [original](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) TF model in two steps,
by adapting instructions from [Google's blog](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193).

1. [Creating TFLite graph from TF checkpoint](#step_1)
1. [Creating TFLite model from TFLite graph](#step_2)
    1. [with the postprocessing layer](#step_2_option_1)
    1. [without the postprocessing layer](#step_2_option_2)
1. [Reference accuracy](#accuracy)

<a name="step_1"></a>
## Step 1: `model.ckpt.*` to `tflite_graph.pb`

We tested this step with TensorFlow v1.11-v1.13, either prebuilt or built from source.

**NB:** On 25/Apr/2019 we informed Google of a bug in their converter, which can be fixed e.g. as follows:
```
anton@diviniti:~/CK_TOOLS/tensorflowmodel-api-master/models/research$ git diff
diff --git a/research/object_detection/export_tflite_ssd_graph.py b/research/object_detection/export_tflite_ssd_graph.py
index b7ed428..1b52335 100644
--- a/research/object_detection/export_tflite_ssd_graph.py
+++ b/research/object_detection/export_tflite_ssd_graph.py
@@ -136,7 +136,7 @@ def main(argv):
   export_tflite_ssd_graph_lib.export_tflite_graph(
       pipeline_config, FLAGS.trained_checkpoint_prefix, FLAGS.output_directory,
       FLAGS.add_postprocessing_op, FLAGS.max_detections,
-      FLAGS.max_classes_per_detection, FLAGS.use_regular_nms)
+      FLAGS.max_classes_per_detection, FLAGS.detections_per_class, FLAGS.use_regular_nms)


 if __name__ == '__main__':
```
This was [fixed upstream](https://github.com/tensorflow/models/commit/9bbf8015dba2133ab2343ec6d6b5096033504e36#r34117181) on 31/May/2019 (albeit in a somewhat less elegant way).

### Manual instructions

#### Install TensorFlow
```
$ python -m pip install tensorflow --user
$ python -c "import tensorflow as tf; print(tf.__version__)"
1.13.1
```

#### Install [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
```
$ export TF_MODEL_API=...
```

#### Install [TensorFlow SSD-MobileNet model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
```bash
$ export $TMP_DIR=/tmp && cd ${TMD_DIR}
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ tar xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ export TF_MODEL_DIR=${PWD}/ssd_mobilenet_v1_coco_2018_01_28
$ ls -la ${TF_MODEL_DIR}
total 58176
drwxr-xr-x  3 anton anton     4096 Feb  1  2018 .
drwxrwxrwt 18 root  root     36864 Apr 26 11:42 ..
-rw-r--r--  1 anton anton       77 Feb  1  2018 checkpoint
-rw-r--r--  1 anton anton 29103956 Feb  1  2018 frozen_inference_graph.pb
-rw-r--r--  1 anton anton 27380740 Feb  1  2018 model.ckpt.data-00000-of-00001
-rw-r--r--  1 anton anton     8937 Feb  1  2018 model.ckpt.index
-rw-r--r--  1 anton anton  3006546 Feb  1  2018 model.ckpt.meta
-rw-r--r--  1 anton anton     4138 Feb  1  2018 pipeline.config
drwxr-xr-x  3 anton anton     4096 Feb  1  2018 saved_model
```

#### Convert
```
$ cd ${TF_MODEL_API}/research
$ export PYTHONPATH=.:./slim:$PYTHONPATH
$ export TFLITE_MODEL_DIR=${TF_MODEL_DIR}
$ python object_detection/export_tflite_ssd_graph.py \
--input_type image_tensor \
--pipeline_config_path ${TF_MODEL_DIR}/pipeline.config \
--trained_checkpoint_prefix ${TF_MODEL_DIR}/model.ckpt \
--output_directory ${TFLITE_MODEL_DIR} \
--add_postprocessing_op=true \
--use_regular_nms=true \
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

### Semi-automated instructions

#### Install TensorFlow
```bash
$ ck install package --tags=lib,tensorflow,v1.13,vcpu

More than one package or version found:

 0) lib-tensorflow-1.13.1-src-cpu  Version 1.13.1  (333b554fb5b0e443)
 1) lib-tensorflow-1.13.1-cpu  Version 1.13.1  (88ad16f0bcfb4ae2)

Please select the package to install [ hit return for "0" ]:
```
Option 1 is faster, but option 0 can be used for [Step 2](#step_2) (where source code is needed).

#### Install [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
```bash
$ ck install package --tags=model,tensorflow,api
```

#### Install [TensorFlow SSD-MobileNet model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
```bash
$ ck install package --tags=model,tensorflow,mlperf,ssd-mobilenet,non-quantized
```

#### Descend into virtual environments one by one
##### TensorFlow
```bash
$ ck virtual env --tags=lib,tensorflow,v1.13,vcpu
$ ${CK_ENV_COMPILER_PYTHON_FILE} -c "import tensorflow as tf; print(tf.__version__)"
```
**NB:** Using `${CK_ENV_COMPILER_PYTHON_FILE}` should ensure that the same version of
Python that was used to install TensorFlow and its dependencies (e.g. `/usr/bin/python3.6`)
will be used to run the conversion script.

##### TensorFlow Object Detection API
```
$ ck virtual env --tags=model,tensorflow,api
$ echo ${CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR}
/home/anton/CK_TOOLS/tensorflowmodel-api-master/models/research/object_detection
```

##### TensorFlow SSD-MobileNet model
```
$ ck virtual env --tags=model,tensorflow,mlperf,ssd-mobilenet,non-quantized
$ echo ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE}
/home/anton/CK_TOOLS/model-tf-mlperf-ssd-mobilenet/model.ckpt
$ echo "$(dirname ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE})"
/home/anton/CK_TOOLS/model-tf-mlperf-ssd-mobilenet
```
**TODO:** Need to introduce an environment variable for the model directory,
so not having to use the `$(dirname ...)` idiom all the time.

#### Convert
```
$ ${CK_ENV_COMPILER_PYTHON_FILE} \
${CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR}/export_tflite_ssd_graph.py \
--input_type image_tensor \
--pipeline_config_path "$(dirname ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE})"/pipeline.config \
--trained_checkpoint_prefix "$(dirname ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE})"/model.ckpt \
--output_directory "$(dirname ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE})" \
--add_postprocessing_op=true \
--use_regular_nms=true \
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
$ grep -A 3 use_regular_nms "$(dirname ${CK_ENV_TENSORFLOW_MODEL_WEIGHTS_FILE})"/tflite_graph.pbtxt
    key: "use_regular_nms"
    value {
      b: true
    }
```

<a name="step_2"></a>
## Step 2: from `tflite_graph.pb` to `detect*.tflite`

**TODO:** Update with manual and semi-automatic instructions.

We tested this step with (the source of) TensorFlow v1.11-v1.13 and Bazel v0.20.0.

<a name="step_2_option_1"></a>
### Option 1: from `tflite_graph.pb` to `detect.tflite`

```bash
$ bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=${TFLITE_MODEL_DIR}/tflite_graph.pb \
    --output_file=${TFLITE_MODEL_DIR}/detect.tflite \
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

<a name="step_2_option_2"></a>
### Option 2: from `tflite_graph.pb` to `detect_cut.tflite`

```bash
$ bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=${TFLITE_MODEL_DIR}/tflite_graph.pb \
    --output_file=${TFLITE_MODEL_DIR}/detect_cut.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='raw_outputs/box_encodings','raw_outputs/class_predictions' \
    --inference_type=FLOAT \
    --allow_custom_ops
```


<a name="accuracy"></a>
## Reference accuracy

### Regular NMS
```
$ ck benchmark program:object-detection-tflite --env.USE_NMS=regular \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
...
Summary:
-------------------------------
Graph loaded in 0.000000s
All images loaded in 0.000000s
All images detected in 0.000000s
Average detection time: 0.000000s
mAP: 0.22349680978666922
Recall: 0.2550505369422975
--------------------------------
```

### Fast NMS
```
$ ck benchmark program:object-detection-tflite --env.USE_NMS=fast \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
...
Summary:
-------------------------------
Graph loaded in 0.000000s
All images loaded in 0.000000s
All images detected in 0.000000s
Average detection time: 0.000000s
mAP: 0.21859688835124763
Recall: 0.24801510024502602
--------------------------------
```

### Fast NMS graph with custom model settings

You can reproduce the regular NMS behaviour even with the fast NMS graph by requesting to use
custom model settings:
```
$ ck benchmark program:object-detection-tflite --env.USE_NMS=fast --env.CUSTOM_MODEL_SETTINGS=yes \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
...
Summary:
-------------------------------
Graph loaded in 0.000000s
All images loaded in 0.000000s
All images detected in 0.000000s
Average detection time: 0.000000s
mAP: 0.22349680978666922
Recall: 0.2550505369422975
--------------------------------
```

For example, you can lower the NMS score threshold for more accurate albeit slower detection:
```
$ ck benchmark program:object-detection-tflite --env.USE_NMS=fast \
--env.CUSTOM_MODEL_SETTINGS=yes --env.NMS_SCORE_THRESHOLD=0.00001 \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

See [here](https://github.com/ctuning/ck-tensorflow/tree/master/program/object-detection-tflite#custom_model_settings) for more details on custom model settings.
