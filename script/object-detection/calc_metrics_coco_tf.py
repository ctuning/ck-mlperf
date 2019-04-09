#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

import numpy as np

from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation

import ck_utils

gt_field = standard_fields.InputDataFields
det_field = standard_fields.DetectionResultFields

def load_groundtruth(file_path, class_name_to_id_map):
  boxes = []
  classes = []
  with open(file_path, 'r') as f:
    for line in f:
      gt = ck_utils.Groundtruth(line)
      boxes.append([gt.x1, gt.y1, gt.x2, gt.y2])
      classes.append(class_name_to_id_map[gt.class_name])
  if boxes:
    return {
      gt_field.groundtruth_boxes: np.array(boxes),
      gt_field.groundtruth_classes: np.array(classes)
    }


def load_detections(file_path):
  boxes = []
  scores = []
  classes = []
  with open(file_path, 'r') as f:
    f.readline() # first line is image size
    for line in f:
      det = ck_utils.Detection(line)
      boxes.append([det.x1, det.y1, det.x2, det.y2])
      scores.append(det.score)
      classes.append(det.class_id)
  # Detections dict can't be empty even there is not detection for this image
  if not boxes:
    boxes = [[0,0,0,0]]
    scores = [0]
    classes = [0]
  return {
    det_field.detection_boxes: np.array(boxes),
    det_field.detection_scores: np.array(scores),
    det_field.detection_classes: np.array(classes)
  }


def evaluate(categories_list, results_dir, txt_annotatins_dir, full_report):
  '''
  Calculate COCO metrics via evaluator class included in TF models repository
  https://github.com/tensorflow/models/tree/master/research/object_detection/metrics

  This method uses annotation converted to txt files.
  This convertation is done by installation dataset-coco-2014 package.
  '''
  class_name_to_id_map = {}
  for category in categories_list:
    # Converted txt annotation lacks spaces in class names
    # and we have to remove spaces from labelmap's class names too
    # to be able to find class id by class name from annotation
    class_name = category['name'].split()
    class_name_no_spaces = ''.join(class_name)
    class_name_to_id_map[class_name_no_spaces] = category['id']
  
  evaluator = coco_evaluation.CocoDetectionEvaluator(categories_list)

  total_dets_count = 0
  total_gts_count = 0
  not_found_gts = []

  files = ck_utils.get_files(results_dir)
  for file_index, file_name in enumerate(files):
    if full_report:
      print('Loading detections and annotations for {} ({} of {}) ...'.format(file_name, file_index+1, len(files)))
    elif (file_index+1) % 100 == 0:
      print('Loading detections and annotations: {} of {} ...'.format(file_index+1, len(files)))

    gt_file = os.path.join(txt_annotatins_dir, file_name)
    det_file = os.path.join(results_dir, file_name)

    # Skip files for which there is no groundtruth
    # e.g. COCO_val2014_000000013466.jpg
    gts = load_groundtruth(gt_file, class_name_to_id_map)
    if not gts:
      not_found_gts.append(file_name)
      continue 

    dets = load_detections(det_file)

    gts_count = gts[gt_field.groundtruth_boxes].shape[0]
    dets_count = dets[det_field.detection_boxes].shape[0]
    total_gts_count += gts_count
    total_dets_count += dets_count

    if full_report:
      print('  Detections: {}'.format(dets_count))
      print('  Groundtruth: {}'.format(gts_count))

    # Groundtruth should be added first, as adding image checks if there is groundtrush for it
    evaluator.add_single_ground_truth_image_info(image_id=file_name, groundtruth_dict=gts)
    evaluator.add_single_detected_image_info(image_id=file_name, detections_dict=dets)

  all_metrics = evaluator.evaluate()

  if not_found_gts:
    print('Groundtrush not found for {} results:'.format(len(not_found_gts)))
    for file_name in not_found_gts:
      print('    {}'.format(file_name))

  print('Total detections: {}'.format(total_dets_count))
  print('Total groundtruths: {}'.format(total_gts_count))
  print('Detection rate: {}'.format(float(total_dets_count)/float(total_gts_count)))
  
  mAP = all_metrics['DetectionBoxes_Precision/mAP']
  recall = all_metrics['DetectionBoxes_Recall/AR@100']
  return mAP, recall, all_metrics
