#
# Copyright (c) 2020 dividiti.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#  

import os


def ck_preprocess(i):

  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)

  inferencepath = dep_env('mlperf-inference-dividiti-rnnt', 'CK_ENV_MLPERF_INFERENCE')

  pythonpath = os.path.join(os.environ['PYTHONPATH'] + ":" + \
                    os.path.join(inferencepath,"v0.7/speech_recognition/rnnt") + ":" + \
                    os.path.join(inferencepath,"v0.7/speech_recognition/rnnt/pytorch"))

  os.environ['PYTHONPATH'] = pythonpath

  return {'return': 0}
