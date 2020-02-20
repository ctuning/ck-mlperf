#
# Copyright (c) 2019 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json
import re
from subprocess import check_output

MLPERF_LOG_ACCURACY_JSON = 'mlperf_log_accuracy.json'
MLPERF_LOG_DETAIL_TXT    = 'mlperf_log_detail.txt'
MLPERF_LOG_SUMMARY_TXT   = 'mlperf_log_summary.txt'
MLPERF_LOG_TRACE_JSON    = 'mlperf_log_trace.json'


def ck_postprocess(i):
  print('\n--------------------------------')

  env               = i['env']
  SIDELOAD_JSON     = env.get('CK_LOADGEN_SIDELOAD_JSON', '')

  save_dict = {}

  # Save logs.
  save_dict['mlperf_log'] = {}
  mlperf_log_dict = save_dict['mlperf_log']

  with open(MLPERF_LOG_ACCURACY_JSON, 'r') as accuracy_file:
    mlperf_log_dict['accuracy'] = json.load(accuracy_file)

  with open(MLPERF_LOG_SUMMARY_TXT, 'r') as summary_file:
    unstripped_summary_lines = summary_file.readlines()
    mlperf_log_dict['summary'] = unstripped_summary_lines

    save_dict['parsed_summary'] = {}
    parsed_summary = save_dict['parsed_summary']
    for line in unstripped_summary_lines:
      pair = line.strip().split(': ', 1)
      if len(pair)==2:
        parsed_summary[ pair[0].strip() ] = pair[1].strip()

  with open(MLPERF_LOG_DETAIL_TXT, 'r') as detail_file:
    mlperf_log_dict['detail'] = detail_file.readlines()

  if os.stat(MLPERF_LOG_TRACE_JSON).st_size==0:
    mlperf_log_dict['trace'] = {}
  else:
    with open(MLPERF_LOG_TRACE_JSON, 'r') as trace_file:
      mlperf_log_dict['trace'] = json.load(trace_file)


  # Check accuracy in accuracy mode.
  accuracy_mode = False
  if mlperf_log_dict['accuracy'] != []:
    accuracy_mode = True

  if accuracy_mode:
    deps = i['deps']
    accuracy_script = os.path.join( deps['mlperf-inference-src']['dict']['env']['CK_ENV_MLPERF_INFERENCE_V05'],
                                    'classification_and_detection', 'tools', 'accuracy-coco.py' )

    coco_dir = deps['dataset']['dict']['deps']['dataset-source']['dict']['env']['CK_ENV_DATASET_COCO']

    for python_dep_name in ('lib-python-numpy', 'tool-coco', 'lib-python-matplotlib'):
        os.environ['PYTHONPATH'] = deps[python_dep_name]['dict']['env']['PYTHONPATH'].split(':')[0] +':'+os.environ.get('PYTHONPATH','')

    command = [ deps['python']['dict']['env']['CK_ENV_COMPILER_PYTHON_FILE'], accuracy_script,
              '--coco-dir', coco_dir,
              '--mlperf-accuracy-file', MLPERF_LOG_ACCURACY_JSON,
    ]

    output = check_output(command).decode('ascii')

    print(output)
    
    searchObj = re.search('mAP=(.+)%', output)

    save_dict['mAP'] = float( searchObj.group(1) )
    
  # for scenario in [ 'SingleStream', 'MultiStream', 'Server', 'Offline' ]:
  #   scenario_key = 'TestScenario.%s' % scenario
  #   scenario = save_dict['results'].get(scenario_key, None)
  #   if scenario: # FIXME: Assumes only a single scenario is valid.
  #     save_dict['execution_time_s']  = scenario.get('took', 0.0)
  #     save_dict['execution_time_ms'] = scenario.get('took', 0.0) * 1000
  #     save_dict['percentiles'] = scenario.get('percentiles', {})
  #     save_dict['qps'] = scenario.get('qps', 0)
  #     if accuracy_mode:
  #       ck.out('mAP=%.3f%% (from the results for %s)' % (scenario.get('mAP', 0.0) * 100.0, scenario_key))

  # save_dict['execution_time'] = save_dict['execution_time_s']


  if SIDELOAD_JSON:
    if os.path.exists(SIDELOAD_JSON):
      with open(SIDELOAD_JSON, 'r') as sideload_fd:
        sideloaded_data = json.load(sideload_fd)
    else:
        sideloaded_data = {}
    save_dict['sideloaded_data'] = sideloaded_data


  with open('tmp-ck-timer.json', 'w') as save_file:
    json.dump(save_dict, save_file, indent=2, sort_keys=True)

  print('--------------------------------\n')
  return {'return': 0}

