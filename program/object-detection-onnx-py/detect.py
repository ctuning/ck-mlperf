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
import onnxruntime as rt

from coco_helper import (load_preprocessed_batch, image_filenames, original_w_h,
    class_labels, num_classes, bg_class_offset, class_map,
    MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE, MODEL_USE_DLA,
    MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_CHANNELS,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE, BATCH_COUNT)


## Model properties:
#
MODEL_PATH = os.environ['CK_ENV_ONNX_MODEL_ONNX_FILEPATH']
INPUT_LAYER_NAME = os.environ['CK_ENV_ONNX_MODEL_INPUT_LAYER_NAME']
OUTPUT_LAYER_BBOXES = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_BBOXES']
OUTPUT_LAYER_LABELS = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_LABELS']
OUTPUT_LAYER_SCORES = os.environ['CK_ENV_ONNX_MODEL_OUTPUT_LAYER_SCORES']


# Program parameters
SCORE_THRESHOLD = float(os.getenv('CK_DETECTION_THRESHOLD', 0.0))
CPU_THREADS = int(os.getenv('CK_HOST_CPU_NUMBER_OF_PROCESSORS',0))


## Writing the results out:
#
CUR_DIR = os.getcwd()
DETECTIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_DETECTIONS_OUT_DIR'])
ANNOTATIONS_OUT_DIR = os.path.join(CUR_DIR, os.environ['CK_ANNOTATIONS_OUT_DIR'])
FULL_REPORT = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')
TIMER_JSON = 'tmp-ck-timer.json'
ENV_JSON = 'env.json'


def main():
    global INPUT_LAYER_NAME
    OPENME = {}

    setup_time_begin = time.time()

    # Load the ONNX model from file
    sess_options = rt.SessionOptions()
    # sess_options.session_log_verbosity_level = 0
    if CPU_THREADS > 0:
        sess_options.enable_sequential_execution = False
        sess_options.session_thread_pool_size = CPU_THREADS
    graph_load_time_begin = time.time()
    sess = rt.InferenceSession(MODEL_PATH, sess_options)
    graph_load_time = time.time() - graph_load_time_begin

    input_layer_names = [x.name for x in sess.get_inputs()]     # FIXME: check that INPUT_LAYER_NAME belongs to this list
    INPUT_LAYER_NAME = INPUT_LAYER_NAME or input_layer_names[0]

    output_layer_names = [x.name for x in sess.get_outputs()]    # FIXME: check that OUTPUT_LAYER_NAME belongs to this list

    model_input_shape = sess.get_inputs()[0].shape
    model_input_type  = sess.get_inputs()[0].type
    model_input_type  = np.uint8 if model_input_type == 'tensor(uint8)' else np.float32     # FIXME: there must be a more humane way!

        # a more portable way to detect the number of classes
    for output in sess.get_outputs():
        if output.name == OUTPUT_LAYER_LABELS:
            model_classes = output.shape[1]


    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print("Input layers: {}".format(input_layer_names))
    print("Output layers: {}".format(output_layer_names))
    print("Input layer name: " + INPUT_LAYER_NAME)
    print("Expected input shape: {}".format(model_input_shape))
    print("Expected input type: {}".format(model_input_type))
    print("Output layer names: " + ", ".join([OUTPUT_LAYER_BBOXES, OUTPUT_LAYER_LABELS, OUTPUT_LAYER_SCORES]))
    print("Data normalization: {}".format(MODEL_NORMALIZE_DATA))
    print("Background/unlabelled classes to skip: {}".format(bg_class_offset))
    print("")

    try:
        expected_batch_size = int(model_input_shape[0])
        if BATCH_SIZE!=expected_batch_size:
            raise Exception("expected_batch_size={}, desired CK_BATCH_SIZE={}, they do not match - exiting.".format(expected_batch_size, BATCH_SIZE))
    except ValueError:
        max_batch_size = None

    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    total_load_time = 0
    next_batch_offset = 0
    total_inference_time = 0
    first_inference_time = 0
    images_loaded = 0

    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1

        begin_time = time.time()
        current_batch_offset = next_batch_offset
        batch_data, next_batch_offset = load_preprocessed_batch(image_filenames, current_batch_offset)

        load_time = time.time() - begin_time
        total_load_time += load_time
        images_loaded += BATCH_SIZE

        # Detect batch
        begin_time = time.time()
        run_options = rt.RunOptions()
        # run_options.run_log_verbosity_level = 0
        batch_results = sess.run([OUTPUT_LAYER_BBOXES, OUTPUT_LAYER_LABELS, OUTPUT_LAYER_SCORES], {INPUT_LAYER_NAME: batch_data}, run_options)
        inference_time = time.time() - begin_time

        print("[batch {} of {}] loading={:.2f} ms, inference={:.2f} ms".format(
            batch_number, BATCH_COUNT, load_time*1000, inference_time*1000))

        total_inference_time += inference_time
        # Remember first batch prediction time
        if batch_index == 0:
            first_inference_time = inference_time

        # Process results
        for index_in_batch in range(BATCH_SIZE):
            global_image_index = current_batch_offset + index_in_batch
            width_orig, height_orig = original_w_h[global_image_index]

            filename_orig = image_filenames[global_image_index]
            detections_filename = os.path.splitext(filename_orig)[0] + '.txt'
            detections_filepath = os.path.join(DETECTIONS_OUT_DIR, detections_filename)
            with open(detections_filepath, 'w') as f:
                f.write('{:d} {:d}\n'.format(width_orig, height_orig))
                for i in range(len(batch_results[2][index_in_batch])):
                    confidence = batch_results[2][index_in_batch][i]
                    if confidence > SCORE_THRESHOLD:
                        class_number = int(batch_results[1][index_in_batch][i])
                        if class_map:
                            class_number = class_map[class_number]
                        else:
                            class_number = class_number

                        box = batch_results[0][index_in_batch][i]
                        x1 = box[0] * width_orig
                        y1 = box[1] * height_orig
                        x2 = box[2] * width_orig
                        y2 = box[3] * height_orig
                        class_label = class_labels[class_number - bg_class_offset]
                        f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {} {}\n'.format(x1,
                                                                                    y1,
                                                                                    x2,
                                                                                    y2,
                                                                                    confidence,
                                                                                    class_number,
                                                                                    class_label
                                                                                    )
                                )

    test_time = time.time() - test_time_begin

    if BATCH_COUNT > 1:
        avg_inference_time = (total_inference_time - first_inference_time) / (images_loaded - BATCH_SIZE)
    else:
        avg_inference_time = total_inference_time / images_loaded

    avg_load_time = total_load_time / images_loaded

    # Save processed images ids list to be able to run
    # evaluation without repeating detections (CK_SKIP_DETECTION=YES)
    # with open(IMAGE_LIST_FILE, 'w') as f:
    #    f.write(json.dumps(processed_image_ids))

    OPENME['setup_time_s'] = setup_time
    OPENME['test_time_s'] = test_time
    OPENME['load_images_time_total_s'] = total_load_time
    OPENME['load_images_time_avg_s'] = avg_load_time
    OPENME['prediction_time_total_s'] = total_inference_time
    OPENME['prediction_time_avg_s'] = avg_inference_time
    OPENME['avg_time_ms'] = avg_inference_time * 1000
    OPENME['avg_fps'] = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    run_time_state = {"run_time_state": OPENME}

    with open(TIMER_JSON, 'w') as o:
        json.dump(run_time_state, o, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
