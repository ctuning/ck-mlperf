#!/usr/bin/env python3

#
# NB: onnxruntime needs numpy v 1.16.* (1.15.* and before would crash)

import os
import onnxruntime as rt
import numpy as np
from PIL import Image


model_path          = os.environ['CK_ENV_ONNX_MODEL_ONNX_FILEPATH']
input_layer_name    = os.environ['CK_ENV_ONNX_MODEL_INPUT_LAYER_NAME']
output_layer_name   = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME']
imagenet_path       = os.environ['CK_ENV_DATASET_IMAGENET_VAL']
labels_path         = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']



def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def load_and_resize_image_to_nchw(image_filepath, height, width):
    pillow_img = Image.open(image_filepath).resize((width, height)) # sic! The order of dimensions in resize is (W,H)

    input_data = np.float32(pillow_img)
#    print(np.array(pillow_img).shape)
    nhwc_data = np.expand_dims(input_data, axis=0)
#    print(nhwc_data.shape)
    nchw_data = nhwc_data.transpose(0,3,1,2)
#    print(nchw_data.shape)
    return nchw_data


labels = load_labels(labels_path)

#print("Device: " + rt.get_device())

sess = rt.InferenceSession(model_path)

input_layer_names   = [ x.name for x in sess.get_inputs() ]     # FIXME: check that input_layer_name belongs to this list
input_layer_name    = input_layer_name or input_layer_names[0]

output_layer_names  = [ x.name for x in sess.get_outputs() ]    # FIXME: check that output_layer_name belongs to this list
output_layer_name   = output_layer_name or output_layer_names[0]

model_input_shape   = sess.get_inputs()[0].shape

# assuming it is NCHW model:
(samples, channels, height, width) = model_input_shape

print("Input layers: {}".format([ str(x) for x in sess.get_inputs()]))
print("Output layers: {}".format([ str(x) for x in sess.get_outputs()]))
print("Input layer name: " + input_layer_name)
print("Expected input shape: {}".format(model_input_shape))
print("Output layer name: " + output_layer_name)
print("")

starting_index = 1
batch_size = 5
batch_count = 2

for batch_idx in range(batch_count):
    print("Batch {}/{}:".format(batch_idx+1,batch_count))
    batch_filenames = [ "ILSVRC2012_val_00000{:03d}.JPEG".format(starting_index + batch_idx*batch_size + i) for i in range(batch_size) ]
    unconcatenated_batch_data = []
    for image_filename in batch_filenames:
        image_filepath = imagenet_path + '/' + image_filename
        nchw_data = load_and_resize_image_to_nchw( image_filepath, height, width )
        unconcatenated_batch_data.append( nchw_data )

    batch_data = np.concatenate(unconcatenated_batch_data, axis=0)
    #print(batch_data.shape)

    batch_predictions = sess.run([output_layer_name], {input_layer_name: batch_data})[0]

    for in_batch_idx in range(batch_size):
        softmax_vector = batch_predictions[in_batch_idx]
        top5_indices = list(reversed(softmax_vector.argsort()))[:5]
        print(batch_filenames[in_batch_idx] + ' :')
        for class_idx in top5_indices:
            print("\t{}\t{}".format(softmax_vector[class_idx], labels[class_idx]))
        print("")

