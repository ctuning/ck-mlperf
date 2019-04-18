#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np
import onnxruntime as rt

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_ONNX_MODEL_ONNX_FILEPATH']
INPUT_LAYER_NAME        = os.environ['CK_ENV_ONNX_MODEL_INPUT_LAYER_NAME']
OUTPUT_LAYER_NAME       = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME']
MODEL_DATA_LAYOUT       = os.environ['ML_MODEL_DATA_LAYOUT']
MODEL_IMAGE_HEIGHT      = int(os.environ['CK_ENV_ONNX_MODEL_IMAGE_HEIGHT'])
MODEL_IMAGE_WIDTH       = int(os.environ['CK_ENV_ONNX_MODEL_IMAGE_WIDTH'])

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv("CK_ENV_ONNX_MODEL_NORMALIZE_DATA") in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv("CK_SUBTRACT_MEAN") == "YES"
USE_MODEL_MEAN          = os.getenv("CK_USE_MODEL_MEAN") == "YES"
MODEL_MEAN_VALUE        = np.array([0, 0, 0], dtype=np.float32) # to be populated

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF'))

## Old perprocessor:
#
# IMAGE_DIR               = os.getenv('RUN_OPT_IMAGE_DIR')
# IMAGE_LIST_FILE         = os.getenv('RUN_OPT_IMAGE_LIST')

LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']

## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))

## Writing the results out:
#
RESULTS_DIR             = os.getenv('CK_RESULTS_DIR')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')


def load_preprocessed_batch(image_list, image_index):
    batch_data = []
    for _ in range(BATCH_SIZE):
        img_file = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_file, np.uint8)
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
        img = img.astype(np.float32)

        # Normalize
        if MODEL_NORMALIZE_DATA:
            img = img/127.5 - 1.0

        # Subtract mean value
        if SUBTRACT_MEAN:
            if USE_MODEL_MEAN:
                img = img - MODEL_MEAN_VALUE
            else:
                img = img - np.mean(img)

        # Add img to batch
        batch_data.append( [img] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        #print(nhwc_data.shape)
        return nhwc_data, image_index
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print(nchw_data.shape)
        return nchw_data, image_index


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def main():
    global INPUT_LAYER_NAME
    global OUTPUT_LAYER_NAME

    print('Images dir: ' + IMAGE_DIR)
    print('Image list file: ' + IMAGE_LIST_FILE)
    print('Model image height: {}'.format(MODEL_IMAGE_HEIGHT))
    print('Model image width: {}'.format(MODEL_IMAGE_WIDTH))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Results dir: ' + RESULTS_DIR);
    print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
    print('Subtract mean: {}'.format(SUBTRACT_MEAN))
    print('Use model mean: {}'.format(USE_MODEL_MEAN))

    labels = load_labels(LABELS_PATH)
    num_labels = len(labels)

    setup_time_begin = time.time()

    # Load preprocessed image filenames:
    with open(IMAGE_LIST_FILE, 'r') as f:
        image_list = [ s.strip() for s in f ]

    # Cleanup results directory
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    # Load the ONNX model from file
    sess = rt.InferenceSession(MODEL_PATH)

    input_layer_names   = [ x.name for x in sess.get_inputs() ]     # FIXME: check that INPUT_LAYER_NAME belongs to this list
    INPUT_LAYER_NAME    = INPUT_LAYER_NAME or input_layer_names[0]

    output_layer_names  = [ x.name for x in sess.get_outputs() ]    # FIXME: check that OUTPUT_LAYER_NAME belongs to this list
    OUTPUT_LAYER_NAME   = OUTPUT_LAYER_NAME or output_layer_names[0]

    model_input_shape   = sess.get_inputs()[0].shape

    if MODEL_DATA_LAYOUT == 'NHWC':
        (samples, height, width, channels) = model_input_shape
    else:
        (samples, channels, height, width) = model_input_shape

    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print("Input layers: {}".format([ str(x) for x in sess.get_inputs()]))
    print("Output layers: {}".format([ str(x) for x in sess.get_outputs()]))
    print("Input layer name: " + INPUT_LAYER_NAME)
    print("Expected input shape: {}".format(model_input_shape))
    print("Output layer name: " + OUTPUT_LAYER_NAME)
    print("Data normalization: {}".format(MODEL_NORMALIZE_DATA))
    print("")

    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    image_index = 0
    load_total_time = 0
    classification_total_time = 0
    images_loaded = 0
    images_processed = 0
    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1
        if FULL_REPORT or (batch_number % 10 == 0):
            print("\nBatch {} of {}".format(batch_number, BATCH_COUNT))
      
        begin_time = time.time()
        batch_data, image_index = load_preprocessed_batch(image_list, image_index)

        load_time = time.time() - begin_time
        load_total_time += load_time
        images_loaded += BATCH_SIZE
        if FULL_REPORT:
            print("Batch loaded in %fs" % (load_time))

        # Classify batch
        begin_time = time.time()
        batch_results = sess.run([OUTPUT_LAYER_NAME], {INPUT_LAYER_NAME: batch_data})[0]
        classification_time = time.time() - begin_time
        if FULL_REPORT:
            print("Batch classified in %fs" % (classification_time))
      
        # Exclude first batch from averaging
        if batch_index > 0 or BATCH_COUNT == 1:
            classification_total_time += classification_time
            images_processed += BATCH_SIZE

        # Process results
        for index_in_batch in range(BATCH_SIZE):
            # Ignore the background class.
            # FIXME: What happens to class 999 (toilet tissue)? Check on
            # e.g. ILSVRC2012_val_00002916.JPEG (74% probability).
            softmax_vector = batch_results[index_in_batch][1:num_labels+1]
            global_index = batch_index * BATCH_SIZE + index_in_batch
            res_file = os.path.join(RESULTS_DIR, image_list[global_index])
            with open(res_file + '.txt', 'w') as f:
                for prob in softmax_vector:
                    f.write('{}\n'.format(prob))
            
    test_time = time.time() - test_time_begin
    classification_avg_time = classification_total_time / images_processed
    load_avg_time = load_total_time / images_loaded


    # Store benchmarking results:
    output_dict = {
        'setup_time_s': setup_time,
        'test_time_s': test_time,
        'images_load_time_s': load_total_time,
        'images_load_time_avg_s': load_avg_time,
        'prediction_time_total_s': classification_total_time,
        'prediction_time_avg_s': classification_avg_time,

        'avg_time_ms': classification_avg_time * 1000,
        'avg_fps': 1.0 / classification_avg_time,
        'batch_time_ms': classification_avg_time * 1000 * BATCH_SIZE,
        'batch_size': BATCH_SIZE,
    }
    with open('tmp-ck-timer.json', 'w') as out_file:
        json.dump(output_dict, out_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
