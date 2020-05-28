#!/usr/bin/env python3

import os
import numpy as np


## Processing in batches:
#
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))


## Model properties:
#
MODEL_IMAGE_HEIGHT      = int(os.getenv('ML_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_HEIGHT',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT',
                              ''))))
MODEL_IMAGE_WIDTH       = int(os.getenv('ML_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_ONNX_MODEL_IMAGE_WIDTH',
                              os.getenv('CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH',
                              ''))))
MODEL_IMAGE_CHANNELS    = int(os.getenv('ML_MODEL_IMAGE_CHANNELS', 3))
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_USE_DLA           = os.getenv('ML_MODEL_USE_DLA', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_MAX_BATCH_SIZE    = int(os.getenv('ML_MODEL_MAX_BATCH_SIZE', BATCH_SIZE))


## Internal processing:
#
INTERMEDIATE_DATA_TYPE  = np.float32    # default for internal conversion
#INTERMEDIATE_DATA_TYPE  = np.int8       # affects the accuracy a bit


## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.fromstring(GIVEN_CHANNEL_MEANS, dtype=np.float32, sep=' ').astype(INTERMEDIATE_DATA_TYPE)
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_MEANS = GIVEN_CHANNEL_MEANS[::-1]     # swapping Red and Blue colour channels


## ImageNet dataset properties:
#
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']


## Preprocessed input images' properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
IMAGE_LIST_FILE_NAME    = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, IMAGE_LIST_FILE_NAME)
IMAGE_DATA_TYPE         = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DATA_TYPE', 'uint8')


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels

class_labels = load_labels(LABELS_PATH)


# Load preprocessed image filenames:
with open(IMAGE_LIST_FILE, 'r') as f:
    image_list = [ s.strip() for s in f ]


def load_preprocessed_batch(image_list, image_index):
    batch_data = []
    for _ in range(BATCH_SIZE):
        img_file = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_file, np.dtype(IMAGE_DATA_TYPE))
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(np.float32)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img, axis=(0,1), keepdims=True)

        if MODEL_INPUT_DATA_TYPE == 'int8' or INTERMEDIATE_DATA_TYPE==np.int8:
            img = np.clip(img, -128, 127).astype(INTERMEDIATE_DATA_TYPE)

        # Add img to batch
        batch_data.append( [img.astype(MODEL_INPUT_DATA_TYPE)] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)
    #print('NHWC shape: {}'.format(nhwc_data.shape))

    if MODEL_DATA_LAYOUT == 'NHWC':
        return nhwc_data, image_index
    elif MODEL_DATA_LAYOUT == 'CHW4':
        dla_batch_padding = MODEL_MAX_BATCH_SIZE-len(batch_data) if MODEL_USE_DLA else 0
        chw4_data = np.pad(nhwc_data, ((0,dla_batch_padding), (0,0), (0,0), (0,1)), 'constant')
        #print('CHW4 shape: {}'.format(chw4_data.shape))
        return chw4_data, image_index
    elif MODEL_DATA_LAYOUT == 'NCHW':
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print('NCHW shape: {}'.format(nchw_data.shape))
        return nchw_data, image_index

