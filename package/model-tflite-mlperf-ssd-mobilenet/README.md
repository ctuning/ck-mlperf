
TODO: Need to move to Zenodo

[Original model link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)

This model was converted from checkpoint format `.ckpt` to tflite protobuf format `.pb`
with Tensorflow 1.11 and [Tensorflow model API](https://github.com/tensorflow/models)
```
python object_detection/export_tflite_ssd_graph.py \
    --input_type image_tensor \
    --pipeline_config_path  <PATH_TO_MODEL>/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config \
    --trained_checkpoint_prefix <PATH_TO_MODEL>/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt \
    --output_directory <PATH_TO_TFLITE_MODEL>/tflite_ssd_mobilenet_v1_coco_2018_01_28 \
    --add_postprocessing_op=true
```

And from tflite protobuf format `.pb` to `.tflite` format with [Tensorflow 1.11](https://github.com/tensorflow/tensorflow)
```
bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=<PATH_TO_TFLITE_MODEL>/tflite_ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb \
    --output_file=<PATH_TO_TFLITE_MODEL>/tflite_ssd_mobilenet_v1_coco_2018_01_28/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=FLOAT \
    --allow_custom_ops
```
and for model without postprocessing layer:
```
bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
--input_file=<PATH_TO_TFLITE_MODEL>/tflite_ssd_mobilenet_v1_coco_2018_01_28/tflite_graph.pb \
--output_file=<PATH_TO_TFLITE_MODEL>/tflite_ssd_mobilenet_v1_coco_2018_01_28/detect_cut.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='raw_outputs/box_encodings','raw_outputs/class_predictions' \
--inference_type=FLOAT \
--allow_custom_ops
```
