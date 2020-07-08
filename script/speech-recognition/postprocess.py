#
# Copyright (c) 2020 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json
import re
#from metrics import word_error_rate


RNNT_INSTRUMENTATION_JSON = 'instr_rnnt.json'

def __gather_predictions(predictions_list: list, labels: list) -> list:
    results = []
    for prediction in predictions_list:
        results += __rnnt_decoder_predictions_tensor(prediction, labels=labels)
    return results


def logits_to_string(logits, labels):
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    return ''.join([labels_map[c] for c in logits])


def ck_postprocess(i):
  print('\n--------------------------------')

  with open(RNNT_INSTRUMENTATION_JSON, 'r') as instr_file:
    instrumentation = json.load(instr_file)

  samples = instrumentation['samples']
  labels = instrumentation['labels']

  for s in samples:
    hypothesis = logits_to_string(s['prediction'], labels)
    reference = logits_to_string(s['transcript'], labels)

    s['hypothesis'] = hypothesis
    s['reference'] = reference

    wer, scores, num_words = 0,0,0
#    wer, scores, num_words = word_error_rate(
#        hypotheses=[prediction], references=[transcript])

    s['wer']=wer
    s['scores']=scores
    s['num_words']=num_words

    s['audio_filepath'] = s['audio_filepath'][0]

    del s['prediction']
    del s['transcript']
    del s['audio_duration']

  instrumentation['samples'] = samples

  with open('tmp-ck-timer.json', 'w') as save_file:
    json.dump(instrumentation, save_file, indent=2, sort_keys=True)

  print('--------------------------------\n')
  return {'return': 0}

