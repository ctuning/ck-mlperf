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


  import ck_utils

  import converter_annotations
  import converter_results


  ENV_INI = 'env.ini'
  ENV = {}

  with open(ENV_INI, 'r') as f:
    for i in f:
      key, value = i.strip().split('=', 1)
      ENV[key] = value
    for key in ['MODEL_IMAGE_WIDTH', 'MODEL_IMAGE_HEIGHT', 'MODEL_IMAGE_CHANNELS', 'IMAGE_COUNT', 'SKIP_IMAGES']:
      ENV[key] = int(ENV[key])
    for key in ['MODEL_NORMALIZE_DATA', 'SAVE_IMAGES', 'FULL_REPORT', 'SKIP_DETECTION', 'VERBOSE']:
      ENV[key] = ENV[key] == 'True'

  ANNOTATIONS_PATH = ENV['ANNOTATIONS_PATH']

  ANNOTATIONS_OUT_DIR = ENV['ANNOTATIONS_OUT_DIR']
  DETECTIONS_OUT_DIR = ENV['DETECTIONS_OUT_DIR']
  IMAGES_OUT_DIR = ENV['IMAGES_OUT_DIR']
  RESULTS_OUT_DIR = ENV['RESULTS_OUT_DIR']

  IMAGE_LIST_FILE = ENV['IMAGE_LIST_FILE']
  LABELMAP_FILE = ENV['LABELMAP_FILE']

  DATASET_TYPE = ENV['DATASET_TYPE']
  METRIC_TYPE = ENV['METRIC_TYPE']
  MODEL_DATASET_TYPE = ENV['MODEL_DATASET_TYPE']

  FULL_REPORT = ENV['FULL_REPORT']
  TIMER_JSON = ENV['TIMER_JSON']

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

  with open(IMAGE_LIST_FILE, 'r') as f:
    processed_image_ids = json.load(f)

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
 
  # Store benchmark results
  with open(TIMER_JSON, 'w') as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  # Print metrics
  print('\nSummary:')
  print('-------------------------------')
  print('Graph loaded in {:.6f}s'.format(OPENME.get('graph_load_time_s', 0)))
  print('All images loaded in {:.6f}s'.format(OPENME.get('images_load_time_s', 0)))
  print('All images detected in {:.6f}s'.format(OPENME.get('detection_time_total_s', 0)))
  print('Average detection time: {:.6f}s'.format(OPENME.get('detection_time_avg_s', 0)))
  print('mAP: {}'.format(OPENME['mAP']))
  print('Recall: {}'.format(OPENME['recall']))
  print('--------------------------------\n')

  return {'return': 0}


if __name__ == "__main__":
  ck_postprocess(0)
