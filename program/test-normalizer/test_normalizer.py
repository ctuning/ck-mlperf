#!/usr/bin/env python3

import time
import os
import numpy as np

from imagenet_helper import (load_preprocessed_batch, image_list, class_labels,
    MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE, MODEL_USE_DLA,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE)

BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))


def main():

    batch_data, image_index = [], 0
    for batch_index in range(BATCH_COUNT):
        before_batch_loading = time.time()

        batch_data, image_index = load_preprocessed_batch(image_list, image_index)
        vectored_batch = np.array(batch_data).ravel().astype(MODEL_INPUT_DATA_TYPE)

        loading_time = time.time() - before_batch_loading

        print("{}-Batch {}/{} took {} seconds to load".format(BATCH_SIZE, batch_index, BATCH_COUNT, loading_time))


if __name__ == '__main__':
    main()
