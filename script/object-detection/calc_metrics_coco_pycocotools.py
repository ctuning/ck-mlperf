#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(image_ids_list, results_dir, annotations_file):
  '''
  Calculate COCO metrics via evaluator from pycocotool package.
  MSCOCO evaluation protocol: http://cocodataset.org/#detections-eval

  This method uses original COCO json-file annotations
  and results of detection converted into json file too.
  '''
  cocoGt = COCO(annotations_file)
  cocoDt = cocoGt.loadRes(results_dir)
  cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
  cocoEval.params.imgIds = image_ids_list
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()

  # These are the same names as object returned by CocoDetectionEvaluator has
  all_metrics = {
    "DetectionBoxes_Precision/mAP": cocoEval.stats[0], 
    "DetectionBoxes_Precision/mAP@.50IOU": cocoEval.stats[1], 
    "DetectionBoxes_Precision/mAP@.75IOU": cocoEval.stats[2], 
    "DetectionBoxes_Precision/mAP (small)": cocoEval.stats[3], 
    "DetectionBoxes_Precision/mAP (medium)": cocoEval.stats[4], 
    "DetectionBoxes_Precision/mAP (large)": cocoEval.stats[5], 
    "DetectionBoxes_Recall/AR@1": cocoEval.stats[6], 
    "DetectionBoxes_Recall/AR@10": cocoEval.stats[7], 
    "DetectionBoxes_Recall/AR@100": cocoEval.stats[8], 
    "DetectionBoxes_Recall/AR@100 (small)": cocoEval.stats[9],
    "DetectionBoxes_Recall/AR@100 (medium)": cocoEval.stats[10], 
    "DetectionBoxes_Recall/AR@100 (large)": cocoEval.stats[11]
  }

  mAP = all_metrics['DetectionBoxes_Precision/mAP']
  recall = all_metrics['DetectionBoxes_Recall/AR@100']
  return mAP, recall, all_metrics
