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
# import shutil
import time
# import PIL
import onnxruntime as rt

#import ck_utils

# import tensorflow as tf

# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
# from object_detection.utils import ops as utils_ops

CUR_DIR = os.getcwd()

## Model properties:
#
MODEL_PATH = os.environ['CK_ENV_ONNX_MODEL_ONNX_FILEPATH']
MODEL_ROOT_PATH = os.environ['CK_ENV_ONNX_MODEL_ROOT']
MODEL_NAME = os.environ['CK_ENV_ONNX_MODEL_NAME']
MODEL_FILE = os.environ['CK_ENV_ONNX_MODEL_ONNX_FILENAME']
LABELS_PATH = os.path.join(MODEL_ROOT_PATH, os.environ['CK_ENV_ONNX_MODEL_CLASSES_LABELS'])
INPUT_LAYER_NAME = os.environ['CK_ENV_ONNX_MODEL_INPUT_LAYER_NAME']
OUTPUT_LAYER_BBOXES = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_BBOXES']
OUTPUT_LAYER_LABELS = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_LABELS']
OUTPUT_LAYER_SCORES = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_SCORES']
MODEL_DATA_LAYOUT = os.environ['ML_MODEL_DATA_LAYOUT']
MODEL_IMAGE_HEIGHT = int(os.environ['CK_ENV_ONNX_MODEL_IMAGE_HEIGHT'])
MODEL_IMAGE_WIDTH = int(os.environ['CK_ENV_ONNX_MODEL_IMAGE_WIDTH'])


## Image normalization:
#
MODEL_NORMALIZE_DATA = os.getenv("CK_ENV_ONNX_MODEL_NORMALIZE_DATA") in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN = os.getenv("CK_SUBTRACT_MEAN") == "YES"
USE_MODEL_MEAN = os.getenv("CK_USE_MODEL_MEAN") == "YES"
MODEL_MEAN_VALUE = np.array([0, 0, 0], dtype=np.float32)  # to be populated

## Input image properties:
#
IMAGE_DIR = os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_DIR')
IMAGE_LIST_FILE = os.path.join(IMAGE_DIR, os.getenv('CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF'))


DATASET_TYPE = os.getenv("CK_ENV_DATASET_TYPE")

# Program parameters
## Processing in batches:
#
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
IMAGE_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
SKIP_IMAGES = int(os.getenv('CK_SKIP_IMAGES', 0))
# SAVE_IMAGES = os.getenv("CK_SAVE_IMAGES") == "YES"
# METRIC_TYPE = (os.getenv("CK_METRIC_TYPE") or DATASET_TYPE).lower()
# IMAGES_OUT_DIR = os.path.join(CUR_DIR, "images")
DETECTIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
RESULTS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_RESULTS_OUT_DIR'])
FULL_REPORT = os.getenv('CK_SILENT_MODE') == 'NO'
SKIP_DETECTION = os.getenv('CK_SKIP_DETECTION') == 'YES'
#IMAGE_LIST_FILE = 'processed_images_id.json'
TIMER_JSON = 'tmp-ck-timer.json'
ENV_JSON = 'env.json'

RESULTS_DIR = os.getenv('CK_RESULTS_DIR')


## Writing the results out:
#
FULL_REPORT = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

# def make_tf_config():
#  mem_percent = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33))
#  num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))
#
#  config = tf.ConfigProto()
#  config.gpu_options.allow_growth = True
#  config.gpu_options.allocator_type = 'BFC'
#  config.gpu_options.per_process_gpu_memory_fraction = mem_percent / 100.0
#  if num_processors > 0:
#    config.device_count["CPU"] = num_processors
#  return config


def load_pil_image_into_numpy_array(image):
    # Check if not RGB and convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Conver to NumPy array
    img_data = np.array(image.getdata())
    img_data = img_data.astype(np.uint8)

    # Make batch from single image
    (im_width, im_height) = image.size
    batch_shape = (1, 3, im_height, im_width)
    batch_data = img_data.reshape(batch_shape)
    return batch_data


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def load_preprocessed_file(image_file):
    batch_data = []
    img = np.fromfile(image_file, np.uint8)
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
    nhwc_data = img

    nhwc_data = np.expand_dims(nhwc_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        # print(nhwc_data.shape)
        return nhwc_data
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        # print(nchw_data.shape)
        return nchw_data


def save_detection_txt(image_file, image_size, output_dict, category_index):
    (im_width, im_height) = image_size
    file_name = os.path.splitext(image_file)[0]
    res_file = os.path.join(DETECTIONS_OUT_DIR, file_name) + '.txt'
    with open(res_file, 'w') as f:
        f.write('{:d} {:d}\n'.format(im_width, im_height))
        for i in range(output_dict['num_detections']):
            class_id = output_dict['detection_classes'][i]
            if 'display_name' in category_index[class_id]:
                class_name = category_index[class_id]['display_name']
            else:
                class_name = category_index[class_id]['name']
            y1, x1, y2, x2 = output_dict['detection_boxes'][i]
            score = output_dict['detection_scores'][i]
            f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:d} {}\n' \
                    .format(x1 * im_width, y1 * im_height, x2 * im_width, y2 * im_height, score, class_id, class_name))


# def save_detection_img(image_file, image_np, output_dict, category_index):
#  if not SAVE_IMAGES: return
#
#  vis_util.visualize_boxes_and_labels_on_image_array(
#      image_np,
#      output_dict['detection_boxes'],
#      output_dict['detection_classes'],
#      output_dict['detection_scores'],
#      category_index,
#      use_normalized_coordinates=True,
#      line_thickness=2)
#  image = PIL.Image.fromarray(image_np)
#  image.save(os.path.join(IMAGES_OUT_DIR, image_file))


def detect():
    global INPUT_LAYER_NAME
    OPENME = {}

    #  # Prepare TF config options
    #  tf_config = make_tf_config()

    #  # Prepare directories
    #  ck_utils.prepare_dir(RESULTS_OUT_DIR)
    #  ck_utils.prepare_dir(ANNOTATIONS_OUT_DIR)
    #  ck_utils.prepare_dir(IMAGES_OUT_DIR)
    #  ck_utils.prepare_dir(DETECTIONS_OUT_DIR)

    # Load processing image filenames

    setup_time_begin = time.time()

    # Load preprocessed image filenames:
    with open(IMAGE_LIST_FILE, 'r') as f:
        image_list = [s.strip() for s in f]

    # Load the ONNX model from file
    sess = rt.InferenceSession(MODEL_PATH)

    input_layer_names = [x.name for x in sess.get_inputs()]     # FIXME: check that INPUT_LAYER_NAME belongs to this list
    INPUT_LAYER_NAME = INPUT_LAYER_NAME or input_layer_names[0]

    output_layer_names = [x.name for x in sess.get_outputs()]    # FIXME: check that OUTPUT_LAYER_NAME belongs to this list

    model_input_shape = sess.get_inputs()[0].shape

    model_classes = sess.get_outputs()[0].shape[1]
    labels = load_labels(LABELS_PATH)
    #bg_class_offset = model_classes-len(labels)  # 1 means the labels represent classes 1..1000 and the background class 0 has to be skipped
    bg_class_offset = 1

    if MODEL_DATA_LAYOUT == 'NHWC':
        (samples, height, width, channels) = model_input_shape
    else:
        (samples, channels, height, width) = model_input_shape

    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print("Input layers: {}".format(input_layer_names))
    print("Output layers: {}".format(output_layer_names))
    print("Input layer name: " + INPUT_LAYER_NAME)
    print("Expected input shape: {}".format(model_input_shape))
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

    for image_index in range(BATCH_COUNT):

        if FULL_REPORT or (image_index % 10 == 0):
            print("\nBatch {} of {}".format(image_index, BATCH_COUNT))

        begin_time = time.time()
        file_name, width, height = image_list[image_index].split(";")
        width = float(width)
        height = float(height)
        img_file = os.path.join(IMAGE_DIR, file_name)
        batch_data = load_preprocessed_file(img_file)

        load_time = time.time() - begin_time
        total_load_time += load_time
        images_loaded += 1
        if FULL_REPORT:
            print("Batch loaded in %fs" % load_time)

        # Detect batch
        begin_time = time.time()
        batch_results = sess.run(['bboxes', 'labels', 'scores'], {INPUT_LAYER_NAME: batch_data})
        detection_time = time.time() - begin_time
        if FULL_REPORT:
            print("Batch classified in %fs" % detection_time)

        total_detection_time += detection_time
        # Remember first batch prediction time
        if image_index == 0:
            first_detection_time = detection_time

        # Process results
        # res_name = file.with.some.name.ext -> file.with.some.name.txt
        res_name = ".".join(file_name.split(".")[:-1]) + ".txt"
        res_file = os.path.join(RESULTS_OUT_DIR, res_name)
        with open(res_file, 'w') as f:
            f.write('{} {}\n'.format(width, height))
            for i in range(len(batch_results[2])):
                score = batch_results[2][0][i]
                if score > 0.5:
                    class_num = batch_results[1][0][i] + bg_class_offset
                    class_name = labels[batch_results[1][0][i]]
                    box = batch_results[0][0][i]
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
    OPENME['images_load_time_total_s'] = total_load_time
    OPENME['images_load_time_avg_s'] = avg_load_time
    OPENME['detection_time_total_s'] = total_detection_time
    OPENME['detection_time_avg_s'] = avg_detection_time
    OPENME['avg_time_ms'] = avg_detection_time * 1000
    OPENME['avg_fps'] = 1.0 / avg_detection_time if avg_detection_time > 0 else 0

    with open(TIMER_JSON, 'w') as o:
        json.dump(OPENME, o, indent=2, sort_keys=True)

    return


def main():

    # Print settings
    print("Model name: " + MODEL_NAME)
    print("Model file: " + MODEL_FILE)
    # print("Model label map file: " + LABELMAP_FILE)
    # print("Model is for dataset: " + MODEL_DATASET_TYPE)

    print("Dataset images: " + IMAGE_DIR)
    print("Dataset annotations: " + ANNOTATIONS_OUT_DIR)
    print("Dataset type: " + DATASET_TYPE)

    print('Image count: {}'.format(IMAGE_COUNT))
    print('Results directory: {}'.format(RESULTS_OUT_DIR))
    print("Temporary annotations directory: " + ANNOTATIONS_OUT_DIR)
    print("Detections directory: " + DETECTIONS_OUT_DIR)
    # print("Result images directory: " + IMAGES_OUT_DIR)
    # print('Save result images: {}'.format(SAVE_IMAGES))

    # Create category index
    # category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_FILE, use_display_name=True)
    # categories_list = category_index.values()  # array: [{"id": 88, "name": "teddy bear"}, ...]
    # print('Categories: {}'.format(categories_list))

    # Run detection if needed
    # ck_utils.print_header('Process images')
    if SKIP_DETECTION:
        print('\nSkip detection, evaluate previous results')
    else:
        detect()


if __name__ == '__main__':
    main()
