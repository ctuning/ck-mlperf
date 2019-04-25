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

1. [Creating TFLite graph from TF checkpoint](#step_1)
2. [Creating TFLite model from TFLite graph](#step_2)
    1. [with the postprocessing layer](#step_2_option_1)
    2. [without the postprocessing layer](#step_2_option_2)

<a name="step_1"></a>
### Step 1: `model.ckpt.*` to `tflite_graph.pb`

We tested this step with TensorFlow v1.11-v1.13, either prebuilt or built from source.

#### Manual instructions

```bash
$ cd /tmp
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ tar xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ export TF_MODEL_DIR=/tmp/ssd_mobilenet_v1_coco_2018_01_28
$ export TFLITE_MODEL_DIR=/tmp/ssd_mobilenet_v1_coco_2018_01_28

$ export TF_MODEL_API=...
$ cd ${TF_MODEL_API}/research
$ export PYTHONPATH=.:./slim:$PYTHONPATH
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
**NB:** On 25/Apr/2019 we informed Google of a bug in their converter, which can be fixed e.g. as follows:
```
anton@diviniti:~/CK_TOOLS/tensorflowmodel-api-master/models/research$ git diff
diff --git a/research/object_detection/export_tflite_ssd_graph.py b/research/object_detection/export_tflite_ssd_graph.py
index b7ed428d..300fb744 100644
--- a/research/object_detection/export_tflite_ssd_graph.py
+++ b/research/object_detection/export_tflite_ssd_graph.py
@@ -136,7 +136,7 @@ def main(argv):
   export_tflite_ssd_graph_lib.export_tflite_graph(
       pipeline_config, FLAGS.trained_checkpoint_prefix, FLAGS.output_directory,
       FLAGS.add_postprocessing_op, FLAGS.max_detections,
-      FLAGS.max_classes_per_detection, FLAGS.use_regular_nms)
+      FLAGS.max_classes_per_detection, use_regular_nms=FLAGS.use_regular_nms)
 
 
 if __name__ == '__main__':
```

#### CK instructions

Install TensorFlow e.g.:

```bash
$ ck install package --tags=lib,tensorflow,v1.13,vcpu

More than one package or version found:

 0) lib-tensorflow-1.13.1-src-cpu  Version 1.13.1  (333b554fb5b0e443)
 1) lib-tensorflow-1.13.1-cpu  Version 1.13.1  (88ad16f0bcfb4ae2)

Please select the package to install [ hit return for "0" ]: 
```

Install the [TensorFlow Model API](https://github.com/tensorflow/models):
```bash
$ ck install package --tags=model,tensorflow,api
```
**To be continued...**

<a name="step_2"></a>
### Step 2: from `tflite_graph.pb` to `detect*.tflite`

We tested this step with (the source of) TensorFlow v1.11-v1.13 and Bazel v0.20.0.

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
