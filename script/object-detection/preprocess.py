#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import inspect
import json
import os
import shutil
import sys
import time

# NB: importing numpy, pillow, etc is delayed until we have loaded the PYTHONPATH from deps{}


SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(SCRIPT_DIR)

import ck_utils

OPENME = {}

def save_preprocessed_image(file_name, image_data):
  image_data.tofile(file_name)

## preprocess(category_index):
def preprocess():

  import PIL.Image

  def load_pil_image_into_numpy_array(image, width, height):

      import numpy as np

      # Check if not RGB and convert to RGB
      if image.mode != 'RGB':
        image = image.convert('RGB')

      image = image.resize((width, height), resample=PIL.Image.BILINEAR)

      # Conver to NumPy array
      img_data = np.array(image.getdata())
      img_data = img_data.astype(np.uint8)

      # Make batch from single image
      batch_shape = (1, height, width, 3)
      batch_data = img_data.reshape(batch_shape)
      return batch_data

  # Prepare directories
  ck_utils.prepare_dir(PREPROCESS_OUT_DIR)  # used by the preprocessor
  ck_utils.prepare_dir(DETECTIONS_OUT_DIR)  # used by the main program
  ck_utils.prepare_dir(RESULTS_OUT_DIR)     # used by the postprocessor
  ck_utils.prepare_dir(ANNOTATIONS_OUT_DIR) # used by the postprocessor

   # Load processing image filenames
  image_files = ck_utils.load_image_list(IMAGES_DIR, IMAGE_COUNT, SKIP_IMAGES)

  # Process images
  load_time_total = 0
  images_processed = 0
  preprocessed_list = []
  for file_counter, image_file in enumerate(image_files):
    if FULL_REPORT or (file_counter+1) % 10 == 0:
      print("\nPreprocess image: {} ({} of {})".format(image_file, file_counter+1, len(image_files)))

    # Load image
    load_time_begin = time.time()
    image = PIL.Image.open(os.path.join(IMAGES_DIR, image_file))
    original_width, original_height = image.size
    
    # The array based representation of the image will be used later 
    # in order to prepare the result image with boxes and labels on it.
    image_data = load_pil_image_into_numpy_array(image, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)

    # NOTE: Insert additional preprocessing here if needed
    preprocessed_file_name = os.path.join(PREPROCESS_OUT_DIR, image_file)
    save_preprocessed_image(preprocessed_file_name, image_data)
    preprocessed_list.append([image_file, original_width, original_height])

    load_time = time.time() - load_time_begin
    load_time_total += load_time

    # Exclude first image from averaging
    if file_counter > 0 or IMAGE_COUNT == 1:
      images_processed += 1

  with open(PREPROCESSED_FILES, "w") as f:
    for row in preprocessed_list:
      f.write("{};{};{}\n".format(row[0], row[1], row[2]))

  load_avg_time = load_time_total / len(preprocessed_list)

  OPENME["images_load_time_s"] = load_time_total
  OPENME["images_load_time_avg_s"] = load_avg_time

  with open(TIMER_JSON, "w") as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)


def ck_preprocess(i):
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def set_in_my_env(var): return my_env(var) and my_env(var).lower() in [ 'yes', 'true', 'on', '1' ]
  def set_in_dep_env(dep, var): return dep_env(dep, var) and dep_env(dep, var).lower() in [ 'yes', 'true', 'on', '1' ]
  def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']

  global ANNOTATIONS_OUT_DIR
  global DETECTIONS_OUT_DIR
  global IMAGES_DIR
  global PREPROCESS_OUT_DIR
  global RESULTS_OUT_DIR

  global DATASET_TYPE
  global FULL_REPORT
  global IMAGE_COUNT
  global MODEL_IMAGE_HEIGHT
  global MODEL_IMAGE_WIDTH
  global SKIP_IMAGES
  global TIMER_JSON
  global PREPROCESSED_FILES

  print('\n--------------------------------')

  PYTHONPATH = os.getenv('PYTHONPATH') or ''
  # NB: importing numpy, pillow, etc is delayed until we have loaded the PYTHONPATH from deps{}
  for dep_name in ['lib-python-numpy', 'lib-python-pillow']:
    PYTHONPATH = dep_env(dep_name, 'PYTHONPATH') + ':' + PYTHONPATH

  split_path = set()
  for p in PYTHONPATH.split(":"):
    if p in ["${PYTHONPATH}", "$PYTHONPATH",""]:
      continue
    split_path.add(p)

  sys.path.extend(list(split_path))     # allow THIS SCRIPT to be able to use numpy, pillow, etc.

  TIMER_JSON = my_env('CK_TIMER_FILE')

  PREPROCESSED_FILES  = my_env('CK_PREPROCESSED_FOF_WITH_ORIGINAL_DIMENSIONS')

  PREPROCESS_OUT_DIR  = my_env('CK_PREPROCESSED_OUT_DIR')
  DETECTIONS_OUT_DIR  = my_env('CK_DETECTIONS_OUT_DIR')
  RESULTS_OUT_DIR     = my_env('CK_RESULTS_OUT_DIR')
  ANNOTATIONS_OUT_DIR = my_env('CK_ANNOTATIONS_OUT_DIR')

  # TODO: all weights packages should provide common vars to reveal its 
  # input image size: https://github.com/ctuning/ck-tensorflow/issues/67

  # Model parameters
  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'):
    MODEL_IMAGE_WIDTH = int(dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'))
    if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT'):
      MODEL_IMAGE_HEIGHT = int(dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT'))
    else:
      MODEL_IMAGE_HEIGHT = MODEL_IMAGE_WIDTH
  else:
    return {'return': 1, 'error': 'Only TensorFlow model packages are currently supported.'}

  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_ROOT') \
     and has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE'):
    MODEL_DATASET_TYPE = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")
    MODEL_IMAGE_CHANNELS = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_IMAGE_CHANNELS")
    if not MODEL_IMAGE_CHANNELS:
      MODEL_IMAGE_CHANNELS = 3
    else:
      MODEL_IMAGE_CHANNELS = int(MODEL_IMAGE_CHANNELS)
    MODEL_NORMALIZE_DATA = set_in_dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA")
    MODEL_SUBTRACT_MEAN = set_in_dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_SUBTRACT_MEAN")
    MODEL_NEED_BACKGROUND_CORRECTION = set_in_dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NEED_BACKGROUND_CORRECTION")
  else:
    print("LABELMAP_FILE = ",dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE'))
    print("MODEL_ROOT = ", dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_ROOT"))
    print("MODEL_DATASET_TYPE = ", dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE"))

    return {'return': 1, 'error': 'Only TensorFlow model packages are currently supported.'}

  # Dataset parameters
  IMAGES_DIR = dep_env('dataset', "CK_ENV_DATASET_IMAGE_DIR")
  DATASET_TYPE = dep_env('dataset', "CK_ENV_DATASET_TYPE")

  # TODO: use the above idiom.
  IMAGE_COUNT = my_env("CK_BATCH_COUNT")
  if not IMAGE_COUNT:
    IMAGE_COUNT = 1
  else:
    IMAGE_COUNT = int(IMAGE_COUNT)
  BATCH_SIZE = my_env("CK_BATCH_SIZE")
  if not BATCH_SIZE:
    BATCH_SIZE = 1
  else:
    BATCH_SIZE = int(BATCH_SIZE)
  SKIP_IMAGES = my_env("CK_SKIP_IMAGES")
  if not SKIP_IMAGES:
    SKIP_IMAGES = 0
  else:
    SKIP_IMAGES = int(SKIP_IMAGES)
  METRIC_TYPE = (my_env("CK_METRIC_TYPE") or DATASET_TYPE).lower()

  SKIP_DETECTION  = set_in_my_env("CK_SKIP_DETECTION")  # actually, this is about skipping the preprocessing
  FULL_REPORT     = not set_in_my_env("CK_SILENT_MODE")
  VERBOSE         = set_in_my_env("VERBOSE")

  # Print settings
  print("Model is for dataset: " + MODEL_DATASET_TYPE)

  print("Dataset images: " + IMAGES_DIR)
  print("Dataset type: " + DATASET_TYPE)

  print("Image count: {}".format(IMAGE_COUNT))
  print("Metric type: " + METRIC_TYPE)
  print("Results directory: {}".format(RESULTS_OUT_DIR))
  print("Temporary annotations directory: " + ANNOTATIONS_OUT_DIR)
  print("Detections directory: " + DETECTIONS_OUT_DIR)
  print("Save preprocessed images: {}".format(PREPROCESSED_FILES))

  # Run detection if needed
  ck_utils.print_header("Process images")
  if SKIP_DETECTION:
    print("\nSkip detection, evaluate previous results")
  else:
    preprocess()

  return {
    'return': 0
  }
