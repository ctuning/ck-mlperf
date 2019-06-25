#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

# import sys
import os
import json
import numpy as np
import time
import tensorflow as tf


CUR_DIR = os.getcwd()

## Model properties:
#
MODEL_ROOT_PATH = os.environ['CK_ENV_TENSORFLOW_MODEL_ROOT']
MODEL_NAME = os.environ['CK_ENV_TENSORFLOW_MODEL_NAME']
MODEL_FILE = os.environ['CK_ENV_TENSORFLOW_MODEL_FILENAME']
MODEL_PATH = os.path.join(MODEL_ROOT_PATH, MODEL_FILE)
LABELS_PATH = os.path.join(MODEL_ROOT_PATH, os.environ['CK_ENV_TENSORFLOW_MODEL_CLASSES_LABELS'])
INPUT_LAYER_NAME = os.environ['CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME']
OUTPUT_LAYER_BBOXES = os.environ['CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_BBOXES']
OUTPUT_LAYER_LABELS = os.environ['CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_LABELS']
OUTPUT_LAYER_SCORES = os.environ['CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_SCORES']
MODEL_DATA_LAYOUT = os.environ['ML_MODEL_DATA_LAYOUT']
MODEL_IMAGE_HEIGHT = int(os.environ['CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT'])
MODEL_IMAGE_WIDTH = int(os.environ['CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'])
MODEL_SKIPPED_CLASSES = os.getenv("CK_ENV_TENSORFLOW_MODEL_SKIPS_ORIGINAL_DATASET_CLASSES", None)

if (MODEL_SKIPPED_CLASSES):
    SKIPPED_CLASSES = [int(x) for x in MODEL_SKIPPED_CLASSES.split(",")]
else:
    SKIPPED_CLASSES = None

## Image normalization:
#
# special normalization mode used in https://github.com/mlperf/inference/blob/master/cloud/image_classification/python/dataset.py
#
GUENTHER_NORM = os.getenv("CK_GUENTHER_NORM") in ('YES', 'yes', 'ON', 'on', '1')
# or
#
MODEL_NORMALIZE_DATA = os.getenv("CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN = os.getenv("CK_SUBTRACT_MEAN") in ('YES', 'yes', 'ON', 'on', '1')
USE_MODEL_MEAN = os.getenv("CK_USE_MODEL_MEAN") in ('YES', 'yes', 'ON', 'on', '1')
MODEL_MEAN_VALUE = np.array([0, 0, 0], dtype=np.float32)  # to be populated

## Input image properties:
#
IMAGE_DIR = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR')
IMAGE_LIST_FILE_NAME = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF')
IMAGE_LIST_FILE = os.path.join(IMAGE_DIR, IMAGE_LIST_FILE_NAME)


DATASET_TYPE = os.getenv("CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")

# Program parameters
## Processing in batches:
#
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
IMAGE_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
SKIP_IMAGES = int(os.getenv('CK_SKIP_IMAGES', 0))
SCORE_THRESHOLD = float(os.getenv('CK_DETECTION_THRESHOLD', 0.3))
DETECTIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
RESULTS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_RESULTS_OUT_DIR'])
FULL_REPORT = os.getenv('CK_SILENT_MODE') == 'NO'
SKIP_DETECTION = os.getenv('CK_SKIP_DETECTION') == 'YES'
TIMER_JSON = 'tmp-ck-timer.json'
ENV_JSON = 'env.json'
RESULTS_DIR = os.getenv('CK_RESULTS_DIR')


## Writing the results out:
#
FULL_REPORT = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

def make_tf_config():
  mem_percent = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33))
  num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.per_process_gpu_memory_fraction = mem_percent / 100.0
  if num_processors > 0:
    config.device_count["CPU"] = num_processors
  return config

def get_handles_to_tensors():
  graph = tf.get_default_graph()
  ops = graph.get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  key_list = [
    OUTPUT_LAYER_BBOXES,
    OUTPUT_LAYER_LABELS,
    OUTPUT_LAYER_SCORES
  ]
  for key in key_list:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
  image_tensor = graph.get_tensor_by_name(INPUT_LAYER_NAME + ':0')
  return tensor_dict, image_tensor

def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def load_preprocessed_file(image_file):
    img = np.fromfile(image_file, np.uint8)
    img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
    img = img.astype(np.float32)

    if GUENTHER_NORM and (MODEL_NAME == "MLPerf SSD-Resnet"):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img / 255. - mean
        img = img / std
    else:
        # Normalize
        if MODEL_NORMALIZE_DATA:
            img = img/127.5 - 1.0

        # Subtract mean value
        if SUBTRACT_MEAN:
            if USE_MODEL_MEAN:
                img = img - MODEL_MEAN_VALUE
            else:
                img = img - np.mean(img)

    # img = img[:, :, ::-1]
    # Add img to batch
    nhwc_data = img[:,:,::-1]

    nhwc_data = np.expand_dims(nhwc_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        # print(nhwc_data.shape)
        return nhwc_data
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        # print(nchw_data.shape)
        return nchw_data

def detect():
    global INPUT_LAYER_NAME
    OPENME = {}

    # Prepare TF config options
    tf_config = make_tf_config()

    setup_time_begin = time.time()

    # Load preprocessed image filenames:
    with open(IMAGE_LIST_FILE, 'r') as f:
        image_list = [s.strip() for s in f]
    
    images_total_count = len(image_list)
    first_index = SKIP_IMAGES
    last_index = BATCH_COUNT * BATCH_SIZE + first_index

    if first_index > images_total_count or last_index > images_total_count:
        print('********************************************')
        print('')
        print('DATASET SIZE EXCEEDED !!!')
        print('Dataset size  : {}'.format(images_total_count))
        print('CK_SKIP_IMAGES: {}'.format(SKIP_IMAGES))
        print('CK_BATCH_COUNT: {}'.format(BATCH_COUNT))
        print('CK_BATCH_SIZE : {}'.format(BATCH_SIZE))
        print('')
        print('********************************************')

    image_list = image_list[SKIP_IMAGES: BATCH_COUNT * BATCH_SIZE + SKIP_IMAGES]

    # Local list of processed files
    with open(IMAGE_LIST_FILE_NAME, 'w') as f:
        for line in image_list:
            f.write('{}\n'.format(line))

    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        graph_def = tf.GraphDef()

        begin_time = time.time()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        graph_load_time = time.time() - begin_time
        print('Graph loaded in {:.4f}s'.format(graph_load_time))

        # Get handles to input and output tensors
        tensor_dict, input_tensor = get_handles_to_tensors()

        setup_time = time.time() - setup_time_begin

        labels = load_labels(LABELS_PATH)
        #bg_class_offset = model_classes-len(labels)  # 1 means the labels represent classes 1..1000 and the background class 0 has to be skipped
        bg_class_offset = 1

        print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
        print("Input layers: {}".format(INPUT_LAYER_NAME))
        print("Output layers: {}".format([OUTPUT_LAYER_BBOXES, OUTPUT_LAYER_LABELS, OUTPUT_LAYER_SCORES]))
        print("Input layer name: " + INPUT_LAYER_NAME)
        print("Expected input shape: {}".format(input_tensor.shape))
        print("Expected input type: {}".format(input_tensor.dtype))
        print("Output layer names: " + ", ".join([OUTPUT_LAYER_BBOXES, OUTPUT_LAYER_LABELS, OUTPUT_LAYER_SCORES]))
        print("Data normalization: {}".format(MODEL_NORMALIZE_DATA))
        print("Background/unlabelled classes to skip: {}".format(bg_class_offset))
        print("")

        setup_time = time.time() - setup_time_begin

        # Run batched mode
        test_time_begin = time.time()
        total_load_time = 0
        total_detection_time = 0
        first_detection_time = 0
        images_loaded = 0

        ## Due to error in ONNX Resnet34 model
        class_map = None
        if (SKIPPED_CLASSES):
            class_map = []
            for i in range(len(labels) + bg_class_offset):
                if i not in SKIPPED_CLASSES:
                    class_map.append(i)

        for image_index in range(BATCH_COUNT):

            if FULL_REPORT or (image_index % 10 == 0):
                print("\nBatch {} of {}".format(image_index + 1, BATCH_COUNT))

            begin_time = time.time()
            file_name, width, height = image_list[image_index].split(";")
            width = float(width)
            height = float(height)
            img_file = os.path.join(IMAGE_DIR, file_name)
            if input_tensor.dtype == tf.float32:
                dtype = np.float32
            elif input_tensor.dtype == tf.uint8:
                dtype = np.uint8
            else:
                print("Unsupported input tensor data type: ", input_tensor.dtype)
                quit()

            batch_data = load_preprocessed_file(img_file).astype(dtype)

            load_time = time.time() - begin_time
            total_load_time += load_time
            images_loaded += 1
            if FULL_REPORT:
                print("Batch loaded in %fs" % load_time)

            # Detect batch
            begin_time = time.time()
            feed_dict = {input_tensor: batch_data}
            output_dict = sess.run(tensor_dict, feed_dict)
            detection_time = time.time() - begin_time

            if FULL_REPORT:
                print("Batch detected in %fs" % detection_time)

            total_detection_time += detection_time
            # Remember first batch prediction time
            if image_index == 0:
                first_detection_time = detection_time

            # Process results
            # res_name = file.with.some.name.ext -> file.with.some.name.txt
            res_name = ".".join(file_name.split(".")[:-1]) + ".txt"
            res_file = os.path.join(DETECTIONS_OUT_DIR, res_name)
            with open(res_file, 'w') as f:
                f.write('{:d} {:d}\n'.format(int(width), int(height)))
                for i in range(len(output_dict["detection_scores"][0])):
                    score = output_dict["detection_scores"][0][i]
                    if score > SCORE_THRESHOLD:
                        if class_map:
                            class_num = class_map[int(output_dict["detection_classes"][0][i])]
                        else:
                            class_num = int(output_dict["detection_classes"][0][i]) + bg_class_offset
                        class_name = labels[class_num - bg_class_offset]
                        box = output_dict["detection_bboxes"][0][i]
                        x1 = box[0] * width
                        y1 = box[1] * height
                        x2 = box[2] * width
                        y2 = box[3] * height
                        f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {} {}\n'.format(x1,
                                                                                    y1,
                                                                                    x2,
                                                                                    y2,
                                                                                    score,
                                                                                    class_num,
                                                                                    class_name
                                                                                    )
                                )
                    else:
                        break

    test_time = time.time() - test_time_begin

    if BATCH_COUNT > 1:
        avg_detection_time = (total_detection_time - first_detection_time) / (images_loaded - BATCH_SIZE)
    else:
        avg_detection_time = total_detection_time / images_loaded

    avg_load_time = total_load_time / images_loaded

    # Save processed images ids list to be able to run
    # evaluation without repeating detections (CK_SKIP_DETECTION=YES)
    # with open(IMAGE_LIST_FILE, 'w') as f:
    #    f.write(json.dumps(processed_image_ids))

    OPENME['setup_time_s'] = setup_time
    OPENME['test_time_s'] = test_time
    OPENME['load_images_time_total_s'] = total_load_time
    OPENME['load_images_time_avg_s'] = avg_load_time
    OPENME['prediction_time_total_s'] = total_detection_time
    OPENME['prediction_time_avg_s'] = avg_detection_time
    OPENME['avg_time_ms'] = avg_detection_time * 1000
    OPENME['avg_fps'] = 1.0 / avg_detection_time if avg_detection_time > 0 else 0

    run_time_state = {"run_time_state": OPENME}

    with open(TIMER_JSON, 'w') as o:
        json.dump(run_time_state, o, indent=2, sort_keys=True)

    return


def main():

    # Print settings
    print("Model name: " + MODEL_NAME)
    print("Model file: " + MODEL_FILE)

    print("Dataset images: " + IMAGE_DIR)
    print("Dataset annotations: " + ANNOTATIONS_OUT_DIR)
    print("Dataset type: " + DATASET_TYPE)

    print('Image count: {}'.format(IMAGE_COUNT))
    print('Results directory: {}'.format(RESULTS_OUT_DIR))
    print("Temporary annotations directory: " + ANNOTATIONS_OUT_DIR)
    print("Detections directory: " + DETECTIONS_OUT_DIR)

    # Run detection if needed
    if SKIP_DETECTION:
        print('\nSkip detection, evaluate previous results')
    else:
        detect()


if __name__ == '__main__':
    main()
