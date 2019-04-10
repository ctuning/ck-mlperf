#!/usr/bin/env python3

import tensorflow as tf

def tf2tflite(input_ft_model_filepath, output_tflite_model_filepath, input_layer_names, output_layer_names):
    "Convert from TF to TFLite"

    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(input_tf_model_filepath, input_layer_names, output_layer_names)
    tflite_model = converter.convert()
    open(output_tflite_model_filepath, "wb").write(tflite_model)


if __name__ == '__main__':
    import os

    input_tf_model_filepath         = os.environ['CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH']
    output_tflite_model_filepath    = os.path.join(os.environ['INSTALL_DIR'], os.environ['PACKAGE_NAME'])

    # FIXME: ask the TF model about its input and output layers.
    #
    input_layer_names = [ os.environ['MODEL_INPUT_LAYER_NAME'] ]
    output_layer_names = [ os.environ['MODEL_OUTPUT_LAYER_NAME'] ]

    tf2tflite(input_tf_model_filepath, output_tflite_model_filepath, input_layer_names, output_layer_names)
