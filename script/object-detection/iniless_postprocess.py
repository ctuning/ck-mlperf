#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#


def ck_postprocess(i):
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def set_in_my_env(var): return my_env(var) and my_env(var).lower() in [ 'yes', 'true', 'on', '1' ]
  def set_in_dep_env(dep, var): return dep_env(dep, var) and dep_env(dep, var).lower() in [ 'yes', 'true', 'on', '1' ]
  def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']
  def has_dep(dep): return dep in i['deps']

  import os
  import json
  import sys
  import inspect

  # gain access to other scripts in the same directory as the postprocess.py :
  #
  SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  sys.path.append(SCRIPT_DIR)

  # gain access to the some python dependencies :
  PYTHONPATH = ''
  for dep_name in ['tool-coco', 'lib-python-matplotlib']:
    PYTHONPATH = dep_env(dep_name, 'PYTHONPATH') + ':' + PYTHONPATH

  split_path = set()
  for p in PYTHONPATH.split(":"):
    if p in ["${PYTHONPATH}", "$PYTHONPATH",""]:
      continue
    split_path.add(p)

  sys.path.extend(list(split_path))     # allow THIS SCRIPT to be able to use numpy, pillow, etc.

  # get some parameters directly from the deps' environment:
  #
  MODEL_ROOT = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_ROOT")
  if MODEL_ROOT:
    LABELMAP_FILE       = os.path.join(MODEL_ROOT, dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE') or "")
    MODEL_DATASET_TYPE  = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")
  else:
    MODEL_ROOT          = dep_env('weights', "CK_ENV_ONNX_MODEL_ROOT")
    LABELMAP_FILE       = os.path.join(MODEL_ROOT, dep_env('weights', 'CK_ENV_ONNX_MODEL_CLASSES_LABELS') or "")
    MODEL_DATASET_TYPE  = dep_env('weights', "CK_ENV_ONNX_MODEL_DATASET_TYPE")

  # Annotations can be a directory or a single file, depending on dataset type:
  ANNOTATIONS_PATH      = dep_env('dataset', "CK_ENV_DATASET_ANNOTATIONS")

  TIMER_JSON            = my_env('CK_TIMER_FILE')

  PREPROCESSED_FILES    = dep_env('dataset', 'CK_ENV_DATASET_OBJ_DETECTION_PREPROCESSED_SUBSET_FOF') or my_env('CK_PREPROCESSED_FOF_WITH_ORIGINAL_DIMENSIONS')

  DETECTIONS_OUT_DIR    = my_env('CK_DETECTIONS_OUT_DIR')
  RESULTS_OUT_DIR       = my_env('CK_RESULTS_OUT_DIR')
  ANNOTATIONS_OUT_DIR   = my_env('CK_ANNOTATIONS_OUT_DIR')

  DATASET_TYPE          = dep_env('dataset', "CK_ENV_DATASET_TYPE")
  METRIC_TYPE           = (my_env("CK_METRIC_TYPE") or DATASET_TYPE).lower()

  FULL_REPORT           = not set_in_my_env("CK_SILENT_MODE")


  import ck_utils
  import converter_annotations
  import converter_results

  ck_utils.prepare_dir(RESULTS_OUT_DIR)
  ck_utils.prepare_dir(ANNOTATIONS_OUT_DIR)

  if METRIC_TYPE != ck_utils.COCO:
    import calc_metrics_coco_tf
    import calc_metrics_kitti
    import calc_metrics_oid
    from object_detection.utils import label_map_util
  else:
    import calc_metrics_coco_pycocotools

  def evaluate(processed_image_ids, categories_list):
    # Convert annotations from original format of the dataset
    # to a format specific for a tool that will calculate metrics
    if DATASET_TYPE != METRIC_TYPE:
      print('\nConvert annotations from {} to {} ...'.format(DATASET_TYPE, METRIC_TYPE))
      annotations = converter_annotations.convert(ANNOTATIONS_PATH, 
                                                  ANNOTATIONS_OUT_DIR,
                                                  DATASET_TYPE,
                                                  METRIC_TYPE)
    else: annotations = ANNOTATIONS_PATH

    # Convert detection results from our universal text format
    # to a format specific for a tool that will calculate metrics
    print('\nConvert results to {} ...'.format(METRIC_TYPE))
    results = converter_results.convert(DETECTIONS_OUT_DIR, 
                                        RESULTS_OUT_DIR,
                                        DATASET_TYPE,
                                        MODEL_DATASET_TYPE,
                                        METRIC_TYPE)

    # Run evaluation tool
    print('\nEvaluate metrics as {} ...'.format(METRIC_TYPE))
    if METRIC_TYPE == ck_utils.COCO:
      mAP, recall, all_metrics = calc_metrics_coco_pycocotools.evaluate(processed_image_ids, results, annotations)
    elif METRIC_TYPE == ck_utils.COCO_TF:
      mAP, recall, all_metrics = calc_metrics_coco_tf.evaluate(categories_list, results, annotations, FULL_REPORT)
    elif METRIC_TYPE == ck_utils.OID:
      mAP, _, all_metrics = calc_metrics_oid.evaluate(results, annotations, LABELMAP_FILE, FULL_REPORT)
      recall = 'N/A'

    else:
      raise ValueError('Metrics type is not supported: {}'.format(METRIC_TYPE))

    OPENME['mAP'] = mAP
    OPENME['recall'] = recall
    OPENME['metrics'] = all_metrics

    return

  OPENME = {}

  with open(PREPROCESSED_FILES, 'r') as f:
    processed_image_filenames = [x.split(';')[0] for x in f.readlines()]

  processed_image_ids = [ ck_utils.filename_to_id(image_filename, DATASET_TYPE) for image_filename in processed_image_filenames ]

  if os.path.isfile(TIMER_JSON):
    with open(TIMER_JSON, 'r') as f:
      OPENME = json.load(f)

  # Run evaluation
  ck_utils.print_header('Process results')
  
  if METRIC_TYPE != ck_utils.COCO:
    category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_FILE, use_display_name=True)
    categories_list = category_index.values()
  else:
    categories_list = []

  evaluate(processed_image_ids, categories_list)

  OPENME['frame_predictions'] = converter_results.convert_to_frame_predictions(DETECTIONS_OUT_DIR)
  OPENME['execution_time'] = OPENME['run_time_state'].get('test_time_s',0) + OPENME['run_time_state'].get('setup_time_s',0)

  # Store benchmark results
  with open(TIMER_JSON, 'w') as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  # Print metrics
  print('\nSummary:')
  print('-------------------------------')
  print('All images loaded in {:.6f}s'.format(OPENME['run_time_state'].get('load_images_time_total_s', 0)))
  print('Average image load time: {:.6f}s'.format(OPENME['run_time_state'].get('load_images_time_avg_s', 0)))
  print('All images detected in {:.6f}s'.format(OPENME['run_time_state'].get('prediction_time_total_s', 0)))
  print('Average detection time: {:.6f}s'.format(OPENME['run_time_state'].get('prediction_time_avg_s', 0)))
  print('mAP: {}'.format(OPENME['mAP']))
  print('Recall: {}'.format(OPENME['recall']))
  print('--------------------------------\n')

  return {'return': 0}


if __name__ == "__main__":
  ck_postprocess(0)
