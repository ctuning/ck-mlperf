#!/usr/bin/env python
# coding: utf-8

# # Generate [dividiti](http://dividiti.com)'s submissions to [MLPerf Inference v0.5](https://github.com/mlperf/inference/tree/master/v0.5)

# <a id="overview"></a>
# ## Overview

# This Jupyter notebook covers [dividiti](http://dividiti.com)'s submissions to [MLPerf Inference v0.5](https://github.com/mlperf/inference/tree/master/v0.5). It validates that experimental data obtained via automated, portable and reproducible [Collective Knowledge](http://cknowledge.org) workflows conforms to [General MLPerf Submission Rules](https://github.com/mlperf/policies/blob/master/submission_rules.adoc)
# and [MLPerf Inference Rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc), including runnning the official [`submission_checker.py`](https://github.com/mlperf/inference/blob/master/v0.5/tools/submission/submission-checker.py).

# A live version of this Jupyter Notebook can be viewed [here](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb).

# ## Table of Contents

# 1. [Overview](#overview)
# 1. [Includes](#includes)
# 1. [System templates](#templates)
#   1. [Firefly RK3399](#templates_firefly)
#   1. [Linaro HiKey960](#templates_hikey960)
#   1. [Huawei Mate 10 Pro](#templates_mate10pro)
#   1. [Raspberry Pi 4](#templates_rpi4)
#   1. [HP Z640](#templates_velociti)
#   1. [Default](#templates_default)
# 1. [Systems](#systems)
# 1. [Implementations](#implementations)
# 1. [Get the experimental data](#get)
#   1. [Image Classification - Closed](#get_image_classification_closed)
#   1. [Image Classification - Open](#get_image_classification_open)
#   1. [Object Detection - Open](#get_object_detection_open)
# 1. [Generate the submission checklist](#checklist)
# 1. [Check the experimental data](#check)

# <a id="includes"></a>
# ## Includes

# ### Standard

# In[ ]:


import os
import sys
import json
import re

from pprint import pprint
from shutil import copy2,copytree
from copy import deepcopy


# ### Scientific

# If some of the scientific packages are missing, please install them using:
# ```
# # python3 -m pip install jupyter pandas numpy matplotlib seaborn --user
# ```

# In[ ]:


import pandas as pd
import numpy as np
import subprocess

print ('Pandas version: %s' % pd.__version__)
print ('NumPy version: %s' % np.__version__)


# No need to hardcode e.g. as:
#   sys.path.append('$CK_TOOLS/tool-coco-master-gcc-8.3.0-compiler.python-3.6.10-linux-64/')
# since it gets added to the Python path automatically via the dependency.
from pycocotools.coco import COCO

# No need to hardcode (e.g. as '$CK_TOOLS/dataset-coco-2017-val'),
# since it gets added to the path automatically via the dependency.
coco_dir = os.environ.get('CK_ENV_DATASET_COCO','')
if coco_dir=='':
    print('Error: Path to COCO dataset not defined!')
    exit(1)

# No need to hardcode (e.g. as '$CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt'),
# since it gets added to the path automatically via the dependency.
imagenet_val_file = os.environ.get('CK_CAFFE_IMAGENET_VAL_TXT','')
if imagenet_val_file=='':
    print('Error: Path to ImageNet labels not defined!')
    exit(1)

# ### Collective Knowledge

# If CK is not installed, please install it using:
# ```
# # python -m pip install ck
# ```

# In[ ]:


import ck.kernel as ck
print ('CK version: %s' % ck.__version__)


# For a particular division, are any results present in the repo?
any_open_results = False
any_closed_results = False

# <a id="systems"></a>
# ## Systems


# Load platform_templates from CK SUT entries
#
r = ck.access({'action':'list', 'module_uoa':'sut'})
if r['return']>0:
    print('Error: %s' % r['error'])
    exit(1)
platform_templates = { sut['data_uoa']: sut['meta']['data'] for sut in r['lst'] }


inference_engine_to_printable = {
    'armnn':            'ArmNN',
    'tflite':           'TFLite',
    'tensorrt':         'TensorRT',
    'tensorflow':       'TensorFlow',
    'openvino':         'OpenVINO',
}

backend_to_printable = {
    'neon':             'Neon',
    'opencl':           'OpenCL',
    'ruy':              'ruy',
    'cpu':              'CPU',
    'cuda':             'CUDA',
    'tensorrt':         'TensorRT-static',
    'tensorrt-dynamic': 'TensorRT-dynamic',
    'edgetpu':          'EdgeTPU',
}

system_description_cache = {}

def dump_system_description_dictionary(target_path, submitter_desc, division, platform, inference_engine, inference_engine_version, backend):

    if target_path in system_description_cache:
        return system_description_cache[target_path]

    library_backend = inference_engine + '_' + inference_engine_version + (('-' + backend) if backend else '')
    division_system = division + '-' + platform + '-' + library_backend

    if library_backend == 'tensorflow-v1.14-cpu':
        status = 'RDI'
    elif library_backend == 'tflite-v1.15.0' or library_backend == 'tensorrt-v6.0':
        status = 'unofficial'
    else:
        status = 'available'

    if inference_engine not in inference_engine_to_printable:
        raise Exception("inference_engine '{}' is unknown, please add it to inference_engine_to_printable dictionary".format(inference_engine))

    framework = inference_engine_to_printable[inference_engine] + ' ' + inference_engine_version + \
                (' ({})'.format(backend_to_printable[backend]) if backend else '')
    framework = framework.replace('for.coral', 'for Coral')

    template = deepcopy(platform_templates[platform])
    template.update({
        'division'  : division,
        'submitter' : submitter_desc,
        'status'    : status,
        'framework' : framework,
    })

    if (not library_backend.endswith('edgetpu') and not library_backend.startswith('tensorrt') and not library_backend.startswith('tensorflow') and not library_backend.endswith('opencl')) \
        or library_backend.endswith('cpu'):
        template.update({
            'accelerator_frequency' : '-',
            'accelerator_memory_capacity' : '-',
            'accelerator_memory_configuration': '-',
            'accelerator_model_name' : '-',
            'accelerator_on-chip_memories': '-',
            'accelerators_per_node' : '0',
        })

    if platform == 'xavier' and not library_backend.startswith('tensorrt'):
        template.update({
            'hw_notes': ''
        })

    if inference_engine == 'tflite' and inference_engine_version == 'v2.3.0':
        template.update({
            'sw_notes': template['sw_notes'] + '. Experimental CMake build from post-v2.3.0/pre-v2.4.0 revision.'
        })

    with open(target_path, 'w') as system_description_file:
        json.dump(template, system_description_file, indent=2)

    system_description_cache[target_path] = template

    return template


# <a id="implementations"></a>
# ## Implementations

implementation_cache = {}

def dump_implementation_dictionary(target_path, model_dict, inference_engine, program_name, benchmark):

    if target_path in implementation_cache:
        return implementation_cache[target_path]

    model_install_env = model_dict['cus']['install_env']
    model_env = model_dict['dict']['env']
    model_tags = model_dict['dict']['tags']

    recorded_model_name = model_install_env.get('ML_MODEL_MODEL_NAME')
    recorded_model_retraining = model_install_env.get('ML_MODEL_RETRAINING', 'no')

    ## fetch recorded model data types, if available, guess if unavailable:
    recorded_model_data_type = model_install_env.get('ML_MODEL_DATA_TYPE')
    recorded_model_input_data_type = model_install_env.get('ML_MODEL_INPUT_DATA_TYPE')

    if not recorded_model_data_type:
        if {'non-quantized', 'fp32', 'float', 'float32'} & set(model_tags):
            recorded_model_data_type = 'fp32'
        elif {'uint8'} & set(model_tags): # 'quantized', 'quant',
            recorded_model_data_type = 'uint8' # 'quantized', 'quant',
        elif {'int8'} & set(model_tags):
            recorded_model_data_type = 'int8'
        elif {'quantized', 'quant'} & set(model_tags):  # FIXME! This is a guess at best!
            print("Warning: could not guess the quantized data type - assuming int8 for now!")
            recorded_model_data_type = 'int8'
        else:
            print("Warning: could not guess whether the model is quantized or not - please add tags or attributes")
            recorded_model_data_type = 'fp32'

    if not recorded_model_input_data_type:  # assume the same
        recorded_model_input_data_type = recorded_model_data_type

    ## recorded_model_input_data_type may need translating from NumPy name into MLPerf's vocabulary:
    model_input_type_mapping = {'float32': 'fp32', 'float16': 'fp16' }
    if recorded_model_input_data_type in model_input_type_mapping:
        recorded_model_input_data_type = model_input_type_mapping[recorded_model_input_data_type]

    ## fetching/constructing the URL of the (original) model:
    starting_weights_filename = None
    if 'PACKAGE_URL' not in model_install_env:  # this model is a result of conversion
        model_deps = model_dict['dict']['deps']
        if 'model-source' in model_deps:
            model_install_env = model_deps['model-source']['dict']['customize']['install_env']
        else:
            starting_weights_filename = model_env['CK_ENV_OPENVINO_MODEL_FILENAME'] # assume it was detected

    if not starting_weights_filename:
        starting_weights_filename = model_install_env['PACKAGE_URL'].rstrip('/') + '/' + model_install_env['PACKAGE_NAME']

    recorded_transformation_path = None
    ## figure out the transformation path:
    if program_name.startswith('openvino-loadgen'):
        if recorded_model_name == 'resnet50':
            model_source = 'TF'
        elif recorded_model_name == 'ssd-resnet34':
            model_source = 'ONNX'
        else:
            model_source = 'Unknown'
        recorded_transformation_path = model_source + ' -> OpenVINO (please refer to closed/Intel/calibration/OpenVINO)'
    elif program_name in [ 'image-classification-tflite-loadgen', 'image-classification-armnn-tflite-loadgen' ]:
        if benchmark in ['resnet', 'resnet50']:
            recorded_transformation_path = 'TF -> TFLite'
        else:
            recorded_transformation_path = 'TFLite'
    elif program_name == 'object-detection-tflite-loadgen':
        if benchmark.endswith('-edgetpu'):  # TODO: need a better signal
            recorded_transformation_path = 'TF -> EdgeTPU'
        else:
            recorded_transformation_path = 'TF -> TFLite'

    elif program_name == 'image-classification-tensorrt-loadgen-py':
        if benchmark in ['resnet', 'resnet50']:
            recorded_transformation_path = 'ONNX'
        else:
            recorded_transformation_path = 'TF'
    elif program_name == 'object-detection-tensorrt-loadgen-py':
        if benchmark in ['ssd-small', 'ssd-mobilenet']:
            recorded_transformation_path = 'TF'
        elif benchmark in ['ssd-large', 'ssd-resnet', 'ssd-resnet34']:
            recorded_transformation_path = 'TF'
    elif program_name == 'mlperf-inference-vision':
        recorded_transformation_path = 'None (TensorFlow)'

    if not recorded_transformation_path:
        raise Exception("Don't know how to derive the transformation path of the model for program:{} and benchmark {}".format(program_name, benchmark))

    # Initial model is never supplied in one of these, so there must have been a transformation:
    if inference_engine in ['armnn', 'tensorrt']:
        recorded_transformation_path += ' -> '+inference_engine_to_printable[inference_engine]

    implementation_dictionary = {
        'retraining': recorded_model_retraining,
        'input_data_types': recorded_model_input_data_type,
        'weight_data_types': recorded_model_data_type,
        'starting_weights_filename': starting_weights_filename,
        'weight_transformations': recorded_transformation_path,

    }

    with open(target_path, 'w') as implementation_file:
        json.dump(implementation_dictionary, implementation_file, indent=2)

    implementation_cache[target_path] = implementation_dictionary

    return implementation_dictionary


# In[ ]:


implementation_readmes = {}
implementation_readmes['image-classification-tflite-loadgen'] = """# MLPerf Inference - Image Classification - TFLite

This C++ implementation uses TFLite to run TFLite models for Image Classification on CPUs.

## Links
- [Jupyter notebook](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb)
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-tflite-loadgen).
- [Instructions](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite/README.md).
"""

implementation_readmes['image-classification-armnn-tflite-loadgen'] = """# MLPerf Inference - Image Classification - ArmNN-TFLite

This C++ implementation uses ArmNN with the TFLite frontend to run TFLite models for Image Classification on Arm Cortex CPUs and Arm Mali GPUs.

## Links
- [Jupyter notebook](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb)
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-armnn-tflite-loadgen).
- [Instructions](https://github.com/ARM-software/armnn-mlperf/blob/master/README.md).
"""

implementation_readmes['image-classification-tensorrt-loadgen-py'] = """# MLPerf Inference - Image Classification - TensorRT

This Python implementation uses TensorRT to run models Image Classification on NVIDIA GPUs.

### Links
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-tensorrt-loadgen-py).
"""

implementation_readmes['object-detection-tensorrt-loadgen-py'] = """# MLPerf Inference - Object Detection - TensorRT

This Python implementation uses TensorRT to run models Object Detection on NVIDIA GPUs.

### Links
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/object-detection-tensorrt-loadgen-py).
"""

implementation_readmes['mlperf-inference-vision'] = """# MLPerf Inference - Object Detection - TensorFlow

This Python implementation is the official MLPerf Inference vision application, modified to support other
object detection models and run with TensorRT.

## Links
- [CK wrapper](https://github.com/ctuning/ck-object-detection/tree/master/program/mlperf-inference-vision).
- [vision_with_ck branch in dividiti's fork of mlperf/inference](https://github.com/dividiti/inference/tree/vision_with_ck).
- [Docker image with instructions](https://github.com/ctuning/ck-mlperf/tree/master/docker/mlperf-inference-vision-with-ck.tensorrt.ubuntu-18.04).
- [Jupyter notebook](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb)
"""

implementation_readmes['openvino-loadgen-v0.7-drop'] = """# MLPerf Inference v0.7 - OpenVINO

Please refer to `closed/Intel/code`.
"""

# In[ ]:

def get_program_path(program_name):

    r = ck.access({'action':'find', 'repo_uoa':'*', 'module_uoa':'program', 'data_uoa':program_name})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)

    return r['path']


# In[ ]:


measurements_readmes = {}
task = 'image-classification'
for division_upper in [ 'Closed', 'Open' ]:
    division_lower = division_upper.lower()
    measurements_readmes[division_lower+'-'+task] = '''# MLPerf Inference - {} Division - Image Classification

We performed our measurements using automated, customizable, portable and reproducible
[Collective Knowledge](http://cknowledge.org) workflows. Our workflows automatically
install dependencies (models, datasets, etc.), preprocess input data in the correct way,
and so on.

## CK repositories

As CK is always evolving, it is hard to pin particular revisions of all repositories.

The most relevant repositories and their latest revisions on the submission date (11/Oct/2019):
- [ck-mlperf](https://github.com/ctuning/ck-mlperf) @ [ee77cfd](https://github.com/ctuning/ck-mlperf/commit/ee77cfd3ddfa30739a8c2f483fe9ba83a233a000) (contains programs integrated with LoadGen, model packages and scripts).
- [ck-env](https://github.com/ctuning/ck-env) @ [f9ac337](https://github.com/ctuning/ck-env/commit/f9ac3372cdc82fa46b2839e45fc67848ab4bac03) (contains dataset descriptions, preprocessing methods, etc.)
- [ck-tensorflow](https://github.com/ctuning/ck-tensorflow) @ [eff8bec](https://github.com/ctuning/ck-tensorflow/commit/eff8bec192021162e4a336dbd3e795afa30b7d26) (contains TFLite packages).
- [armnn-mlperf](https://github.com/arm-software/armnn-mlperf) @ [42f44a2](https://github.com/ARM-software/armnn-mlperf/commit/42f44a266b6b4e04901255f46f6d34d12589208f) (contains ArmNN/ArmCL packages).

## Links
- [Bash script](https://github.com/ctuning/ck-mlperf/tree/master/script/mlperf-inference-v0.5.{}.image-classification) used to invoke benchmarking on Linux systems or Android devices.
'''.format(division_upper, division_lower)


task = 'object-detection'
for division_upper in [ 'Closed', 'Open' ]:
    division_lower = division_upper.lower()
    measurements_readmes[division_lower+'-'+task] = '''# MLPerf Inference - {} Division - Object Detection

We performed our measurements using automated, customizable, portable and reproducible
[Collective Knowledge](http://cknowledge.org) workflows. Our workflows automatically
install dependencies (models, datasets, etc.), preprocess input data in the correct way,
and so on.

## CK repositories

As CK is always evolving, it is hard to pin particular revisions of all repositories.

The most relevant repositories and their latest revisions on the submission date (18/Oct/2019):

- [ck-mlperf](https://github.com/ctuning/ck-mlperf) @ [ef1fced](https://github.com/ctuning/ck-mlperf/commit/ef1fcedd495fd03b5ad6d62d62c8ba271854f2ad) (contains the CK program wrapper, MLPerf SSD-MobileNet model packages and scripts).
- [ck-object-detection](https://github.com/ctuning/ck-object-detection) @ [780d328](https://github.com/ctuning/ck-object-detection/commit/780d3288ec19656cb60c5ad39b2486bbf0fbf97a) (contains most model packages)
- [ck-env](https://github.com/ctuning/ck-env) @ [5af9fbd](https://github.com/ctuning/ck-env/commit/5af9fbd93ad6c6465b631716645ad9442a333442) (contains dataset descriptions, preprocessing methods, etc.)

## Links
- [Docker image with instructions](https://github.com/ctuning/ck-mlperf/tree/master/docker/mlperf-inference-vision-with-ck.tensorrt.ubuntu-18.04).
- [Bash script](https://github.com/ctuning/ck-mlperf/tree/master/script/mlperf-inference-v0.5.{}.object-detection) used to invoke benchmarking via the Docker image.
'''.format(division_upper, division_lower)


# In[ ]:


# Snapshot of https://github.com/dividiti/inference/blob/61220457dec221ed1984c62bd9d382698bd71bc6/v0.5/mlperf.conf
mlperf_conf_6122045 = '''
# The format of this config file is 'key = value'.
# The key has the format 'model.scenario.key'. Value is mostly int64_t.
# Model maybe '*' as wildcard. In that case the value applies to all models.
# All times are in milli seconds

*.SingleStream.target_latency = 10
*.SingleStream.target_latency_percentile = 90
*.SingleStream.min_duration = 60000
*.SingleStream.min_query_count = 1024

*.MultiStream.target_qps = 20
*.MultiStream.target_latency_percentile = 99
*.MultiStream.samples_per_query = 4
*.MultiStream.max_async_queries = 1
*.MultiStream.target_latency = 50
*.MultiStream.min_duration = 60000
*.MultiStream.min_query_count = 270336
ssd-resnet34.MultiStream.target_qps = 15
ssd-resnet34.MultiStream.target_latency = 66
gnmt.MultiStream.min_query_count = 90112
gnmt.MultiStream.target_latency = 100
gnmt.MultiStream.target_qps = 10
gnmt.MultiStream.target_latency_percentile = 97

*.Server.target_qps = 1.0
*.Server.target_latency = 10
*.Server.target_latency_percentile = 99
*.Server.target_duration = 0
*.Server.min_duration = 60000
*.Server.min_query_count = 270336
resnet50.Server.target_latency = 15
ssd-resnet34.Server.target_latency = 100
gnmt.Server.min_query_count = 90112
gnmt.Server.target_latency = 250
gnmt.Server.target_latency_percentile = 97

*.Offline.target_qps = 1.0
*.Offline.target_latency_percentile = 90
*.Offline.min_duration = 60000
*.Offline.min_query_count = 1
'''


# <a id="get"></a>
# ## Get the experimental data

# Download experimental data and add CK repositories as follows.

# <a id="get_image_classification_closed"></a>
# ### Image Classification - Closed (MobileNet, ResNet)

# #### `firefly`

# ```
# $ wget https://www.dropbox.com/s/3md826fk7k1taf3/mlperf.closed.image-classification.firefly.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.closed.image-classification.firefly.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/jusoz329mhixpxm/mlperf.closed.image-classification.firefly.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.closed.image-classification.firefly.armnn-v19.08.neon.zip
#
# $ wget https://www.dropbox.com/s/08lzbz7jl2w5jhu/mlperf.closed.image-classification.firefly.armnn-v19.08.opencl.zip
# $ ck add repo --zip=mlperf.closed.image-classification.firefly.armnn-v19.08.opencl.zip
# ```

# #### `hikey960`

# ```
# $ wget https://www.dropbox.com/s/lqnffl6wbaeceul/mlperf.closed.image-classification.hikey960.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.closed.image-classification.hikey960.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/6m6uv1d33yc82f8/mlperf.closed.image-classification.hikey960.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.closed.image-classification.hikey960.armnn-v19.08.neon.zip
#
# $ wget https://www.dropbox.com/s/bz56y4damfqggr8/mlperf.closed.image-classification.hikey960.armnn-v19.08.opencl.zip
# $ ck add repo --zip=mlperf.closed.image-classification.hikey960.armnn-v19.08.opencl.zip
# ```

# #### `rpi4`

# ```
# $ wget https://www.dropbox.com/s/ig97x9cqoxfs3ne/mlperf.closed.image-classification.rpi4.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.closed.image-classification.rpi4.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/ohcuyes409h66tx/mlperf.closed.image-classification.rpi4.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.closed.image-classification.rpi4.armnn-v19.08.neon.zip
# ```

# #### `mate10pro`

# ```
# $ wget https://www.dropbox.com/s/r7hss1sd0268b9j/mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon.zip
#
# $ wget https://www.dropbox.com/s/iflzxbxcv3qka9x/mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl.zip
# $ ck add repo --zip=mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl.zip
# ```

# **NB:** We aborted the ResNet accuracy experiment with TFLite, as it was estimated to take 17 hours.

# #### `mate10pro` (only for testing the checker)

# ##### BAD_LOADGEN

# ```
# $ wget https://www.dropbox.com/s/nts8e7unb7vm68f/mlperf.closed.image-classification.mate10pro.tflite-v1.13.mobilenet.BAD_LOADGEN.zip
# $ ck add repo --zip=mlperf.closed.image-classification.mate10pro.tflite-v1.13.mobilenet.BAD_LOADGEN.zip
# ```

# ##### BAD_RESNET

# ```
# $ wget https://www.dropbox.com/s/bi2owxxpcfm6n2s/mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl.BAD_RESNET.zip
# $ ck add repo --zip=mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl.BAD_RESNET.zip
#
# $ wget https://www.dropbox.com/s/t2o2elqdyitqlpi/mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon.BAD_RESNET.zip
# $ ck add repo --zip=mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon.BAD_RESNET.zip
# ```

# <a id="get_image_classification_open"></a>
# ### Image Classification - Open (MobileNets-v1,v2)

# #### `firefly`

# ```
# $ wget https://www.dropbox.com/s/q8ieqgnr3zn6w4y/mlperf.open.image-classification.firefly.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.open.image-classification.firefly.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/zpenduz1i4qt651/mlperf.open.image-classification.firefly.tflite-v1.15.mobilenet-v1-quantized.zip
# $ ck add repo --zip=mlperf.open.image-classification.firefly.tflite-v1.15.mobilenet-v1-quantized.zip
#
# $ wget https://www.dropbox.com/s/3mmefvxc15m9o5b/mlperf.open.image-classification.firefly.armnn-v19.08.opencl.zip
# $ ck add repo --zip=mlperf.open.image-classification.firefly.armnn-v19.08.opencl.zip
#
# $ wget https://www.dropbox.com/s/hrupp4o4apo3dfa/mlperf.open.image-classification.firefly.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.open.image-classification.firefly.armnn-v19.08.neon.zip
# ```

# #### `hikey960`

# ```
# $ wget https://www.dropbox.com/s/2gbbpsd2pjurvc8/mlperf.open.image-classification.hikey960.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.open.image-classification.hikey960.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/rmttjnxzih9snzh/mlperf.open.image-classification.hikey960.tflite-v1.15.mobilenet-v1-quantized.zip
# $ ck add repo --zip=mlperf.open.image-classification.hikey960.tflite-v1.15.mobilenet-v1-quantized.zip
#
# $ wget https://www.dropbox.com/s/m5illg8i2tse5hg/mlperf.open.image-classification.hikey960.armnn-v19.08.opencl.zip
# $ ck add repo --zip=mlperf.open.image-classification.hikey960.armnn-v19.08.opencl.zip
#
# $ wget https://www.dropbox.com/s/3cujqfe4ps0g66h/mlperf.open.image-classification.hikey960.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.open.image-classification.hikey960.armnn-v19.08.neon.zip
# ```

# #### `rpi4`

# ```
# $ wget https://www.dropbox.com/s/awhdqjq3p4tre2q/mlperf.open.image-classification.rpi4.tflite-v1.15.zip
# $ ck add repo --zip=mlperf.open.image-classification.rpi4.tflite-v1.15.zip
#
# $ wget https://www.dropbox.com/s/rf8vsg5firhjzf8/mlperf.open.image-classification.rpi4.tflite-v1.15.mobilenet-v1-quantized.zip
# $ ck add repo --zip=mlperf.open.image-classification.rpi4.tflite-v1.15.mobilenet-v1-quantized.zip
#
# $ wget https://www.dropbox.com/s/0oketvqml7gyzl0/mlperf.open.image-classification.rpi4.armnn-v19.08.neon.zip
# $ ck add repo --zip=mlperf.open.image-classification.rpi4.armnn-v19.08.neon.zip
# ```

# #### `mate10pro`

# ```
# $ wget https://www.dropbox.com/s/avi6h9m2demz5zr/mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet.zip
# $ ck add repo --zip=mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet.zip
#
# $ wget https://www.dropbox.com/s/soaw27zcjb8hhww/mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet-v1-quantized.zip
# $ ck add repo --zip=mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet-v1-quantized.zip
# ```

# **NB:** `mate10pro.tflite-v1.13.mobilenet` would have been a perfectly valid closed submission, just finished a little bit late after the deadline. `mate10pro.tflite-v1.13.mobilenet-quantized` is an open submission alright, as dividiti hadn't declared submitting quantized results before the deadline.

# <a id="get_object_detection_open"></a>
# ### Object Detection - Open

# #### `velociti`

# ```
# $ wget https://www.dropbox.com/s/wiea3a8zf077jsv/mlperf.open.object-detection.velociti.zip
# $ ck add repo --zip=mlperf.open.object-detection.velociti.zip
# ```

# <a id="checklist"></a>
# ## Generate the submission checklist

# In[ ]:

# See https://github.com/mlperf/inference_policies/issues/170
checklist_template_0_7 = """MLPerf Inference 0.7 Self-Certification Checklist

Name of Certifying Engineer(s): %(name)s

Email of Certifying Engineer(s): %(email)s

Name of System(s) Under Test: %(system_name)s

Does the submission run the same code in accuracy and performance
modes? (check one)
- [x] Yes
- [ ] No

Where is the LoadGen trace stored? (check one)
- [x] Host DRAM
- [ ] Other, please specify:

Are the weights calibrated using data outside of the calibration set?
(check one)
- [ ] Yes
- [x] No

What untimed pre-processing does the submission use? (check all that apply)
- [x] Resize
- [x] Reorder channels or transpose
- [ ] Pad
- [x] A single crop
- [x] Mean subtraction and normalization
- [ ] Convert to whitelisted format
- [ ] No pre-processing
- [ ] Other, please specify:

What numerics does the submission use? (check all that apply)
- [ ] INT4
- [%(numerics_int8)s] INT8
- [ ] INT16
- [ ] INT32
- [%(numerics_uint8)s] UINT8
- [ ] UINT16
- [ ] UINT32
- [ ] FP11
- [%(numerics_fp16)s] FP16
- [ ] BF16
- [%(numerics_fp32)s] FP32
- [ ] Other, please specify:

Which of the following techniques does the submission use? (check all that apply)
- [ ] Wholesale weight replacement
- [ ] Weight supplements
- [ ] Discarding non-zero weight elements
- [ ] Pruning
- [ ] Caching queries
- [ ] Caching responses
- [ ] Caching intermediate computations
- [ ] Modifying weights during the timed portion of an inference run
- [ ] Weight quantization algorithms that are similar in size to the
non-zero weights they produce
- [ ] Hard coding the total number of queries
- [ ] Techniques that boost performance for fixed length experiments but
are inapplicable to long-running services except in the offline
scenario
- [ ] Using knowledge of the LoadGen implementation to predict upcoming
lulls or spikes in the server scenario
- [ ] Treating beams in a beam search differently. For example,
employing different precision for different beams
- [ ] Changing the number of beams per beam search relative to the reference
- [ ] Incorporating explicit statistical information about the performance or accuracy sets
- [ ] Techniques that take advantage of upsampled images.
- [ ] Techniques that only improve performance when there are identical samples in a query.
- [x] None of the above

Is the submission congruent with all relevant MLPerf rules?
- [x] Yes
- [ ] No

For each SUT, does the submission accurately reflect the real-world
performance of the SUT?
- [x] Yes
- [ ] No
"""

def get_checklist_0_7(checklist_template=checklist_template_0_7,
        name='Anton Lokhmotov', email='anton@dividiti.com',
        system_name='Raspberry Pi 4 (rpi4)', numerics='fp32'):
    def tick(var): return "x" if var else " "
    print("=" * 100)
    print(system_name)
    print("=" * 100)
    checklist = checklist_template % {
        "name" : name,
        "email" : email,
        "system_name": system_name,
        "numerics_int8": tick(numerics=='int8'),
        "numerics_uint8": tick(numerics=='uint8'),
        "numerics_fp16": tick(numerics=='fp16'),
        "numerics_fp32": tick(numerics=='fp32'),
    }
    print(checklist)
    print("-" * 100)

    return checklist


checklist_template_0_5 = """MLPerf Inference 0.5 Self-Certification Checklist

Name of Certifying Engineer(s): %(name)s

Email of Certifying Engineer(s): %(email)s

Name of System(s) Under Test: %(system_name)s

Division (check one):
- [%(open)s] Open
- [%(closed)s] Closed

Category (check one):
- [%(category_available)s] Available
- [%(category_preview)s] Preview
- [%(category_rdi)s] Research, Development, and Internal (RDI)

Benchmark (check one):
- [%(benchmark_mobilenet)s] MobileNet
- [ ] SSD-MobileNet
- [%(benchmark_resnet)s] ResNet
- [ ] SSD-1200
- [ ] NMT
- [%(benchmark_other)s] Other, please specify: %(benchmark_other_specify)s

Please fill in the following tables adding lines as necessary:
97%%-tile latency is required for NMT only. 99%%-tile is required for all other models.

### Single Stream Results Table
| SUT Name | Benchmark | Query Count | Accuracy |
|----------|-----------|-------------|----------|
| %(system)s | %(benchmark)s | %(query_count)s | %(accuracy_pc)s%% |

### Multi-Stream Results Table
| SUT Name | Benchmark | Query Count |  Accuracy | 97%%-tile Latency | 99%%-tile Latency |
|----------|-----------|-------------|-----------|------------------|------------------|
|          |           |             |           |                  |                  |

### Server Results Table
| SUT Name | Benchmark | Query Count | Accuracy | 97%%-tile Latency | 99%%-tile Latency |
|----------|-----------|-------------|----------|------------------|------------------|
|          |           |             |          |                  |                  |

### Offline Results Table
| SUT Name | Benchmark | Sample Count | Accuracy |
|----------|-----------|--------------|----------|
|          |           |              |          |

Scenario (check all that apply):
- [%(scenario_singlestream)s] Single-Stream
- [%(scenario_multistream)s] Multi-Stream
- [%(scenario_server)s] Server
- [%(scenario_offline)s] Offline

For each SUT, does the submission meet the latency target for each
combination of benchmark and scenario? (check all that apply)
- [x] Yes (Single-Stream and Offline no requirements)
- [ ] Yes (MobileNet x Multi-Stream 50 ms @ 99%%)
- [ ] Yes (MobileNet x Server 10 ms @ 99%%)
- [ ] Yes (SSD-MobileNet x Multi-Stream 50 ms @ 99%%)
- [ ] Yes (SSD-MobileNet x Server 10 ms @ 99%%)
- [ ] Yes (ResNet x Multi-Stream 50 ms @ 99%%)
- [ ] Yes (ResNet x Server 15 ms @ 99%%)
- [ ] Yes (SSD-1200 x Multi-Stream 66 ms @ 99%%).
- [ ] Yes (SSD-1200 x Server 100 ms @ 99%%)
- [ ] Yes (NMT x Multi-Stream 100 ms @ 97%%)
- [ ] Yes (NMT x Server 250 ms @ 97%%)
- [ ] No


For each SUT, is the appropriate minimum number of queries or samples
met, depending on the Scenario x Benchmark? (check all that apply)
- [x] Yes (Single-Stream 1,024 queries)
- [ ] Yes (Offline 24,576 samples)
- [ ] Yes (NMT Server and Multi-Stream 90,112 queries)
- [ ] Yes (Image Models Server and Multi-Stream 270,336 queries)
- [ ] No

For each SUT and scenario, is the benchmark accuracy target met?
(check all that apply)
- [%(mobilenet_accuracy_met)s] Yes (MobileNet 71.68%% x 98%%)
- [ ] Yes (SSD-MobileNet 0.22 mAP x 99%%)
- [%(resnet_accuracy_met)s] Yes (ResNet 76.46%% x 99%%)
- [ ] Yes (SSD-1200 0.20 mAP x 99%%)
- [ ] Yes (NMT 23.9 BLEU x 99%%)
- [%(accuracy_not_met)s] No

For each SUT and scenario, did the submission run on the whole
validation set in accuracy mode? (check one)
- [x] Yes
- [ ] No

How many samples are loaded into the QSL in performance mode?
%(performance_sample_count)s

For each SUT and scenario, does the number of loaded samples in the
QSL in performance mode meet the minimum requirement?  (check all that
apply)
- [%(performance_sample_count_1024)s] Yes (ResNet and MobileNet 1,024 samples)
- [%(performance_sample_count_256)s] Yes (SSD-MobileNet 256 samples)
- [%(performance_sample_count_64)s] Yes (SSD-1200 64 samples)
- [ ] Yes (NMT 3,903,900 samples)
- [%(performance_sample_count_not_met)s] No

For each SUT and scenario, is the experimental duration greater than
or equal to 60 seconds?  (check one)
- [x] Yes
- [ ] No

Does the submission use LoadGen? (check one)
- [x] Yes
- [ ] No

Is your loadgen commit from one of these allowed commit hashes?
- [%(revision_61220457de)s] 61220457dec221ed1984c62bd9d382698bd71bc6
- [%(revision_5684c11e39)s] 5684c11e3987b614aae830390fa0e92f56b7e800
- [%(revision_55c0ea4e77)s] 55c0ea4e772634107f3e67a6d0da61e6a2ca390d
- [%(revision_d31c18fbd9)s] d31c18fbd9854a4f1c489ca1bc4cd818e48f2bc5
- [%(revision_1d0e06e54a)s] 1d0e06e54a7d763cf228bdfd8b1e987976e4222f
- [%(revision_other)s] Other, please specify: %(revision_other_specify)s

Do you have any additional change to LoadGen? (check one)
- [ ] Yes, please specify:
- [x] No

Does the submission run the same code in accuracy and performance
modes? (check one)
- [x] Yes
- [ ] No

Where is the LoadGen trace stored? (check one)
- [x] Host DRAM
- [ ] Other, please specify:

For the submitted result, what is the QSL random number generator seed?
- [x] 0x2b7e151628aed2a6ULL (3133965575612453542)
- [ ] Other, please specify:

For the submitted results, what is the sample index random number generator seed?
- [x] 0x093c467e37db0c7aULL (665484352860916858)
- [ ] Other, please specify:

For the submitted results, what is the schedule random number generator seed?
- [x] 0x3243f6a8885a308dULL (3622009729038561421)
- [ ] Other, please specify:

For each SUT and scenario, is the submission run the correct number of
times for the relevant scenario? (check one)
- [x] Yes (Accuracy 1x Performance 1x Single-Stream, Multi-Stream, Offline)
- [ ] Yes (Accuracy 1x Performance 5x Server)
- [ ] No

Are the weights calibrated using data outside of the calibration set?
(check one)
- [ ] Yes
- [x] No

What untimed pre-processing does the submission use? (check all that apply)
- [x] Resize
- [ ] Reorder channels or transpose
- [ ] Pad
- [x] A single crop
- [x] Mean subtraction and normalization
- [ ] Convert to whitelisted format
- [ ] No pre-processing
- [ ] Other, please specify:

What numerics does the submission use? (check all that apply)
- [ ] INT4
- [ ] INT8
- [ ] INT16
- [%(numerics_uint8)s] UINT8
- [ ] UINT16
- [ ] FP11
- [ ] FP16
- [ ] BF16
- [%(numerics_fp32)s] FP32
- [ ] Other, please specify:

Which of the following techniques does the submission use? (check all that apply)
- [ ] Wholesale weight replacement
- [ ] Weight supplements
- [ ] Discarding non-zero weight elements
- [ ] Pruning
- [ ] Caching queries
- [ ] Caching responses
- [ ] Caching intermediate computations
- [ ] Modifying weights during the timed portion of an inference run
- [ ] Weight quantization algorithms that are similar in size to the
non-zero weights they produce
- [ ] Hard coding the total number of queries
- [ ] Techniques that boost performance for fixed length experiments but
are inapplicable to long-running services except in the offline
scenario
- [ ] Using knowledge of the LoadGen implementation to predict upcoming
lulls or spikes in the server scenario
- [ ] Treating beams in a beam search differently. For example,
employing different precision for different beams
- [ ] Changing the number of beams per beam search relative to the reference
- [ ] Incorporating explicit statistical information about the performance or accuracy sets
- [ ] Techniques that take advantage of upsampled images.
- [ ] Techniques that only improve performance when there are identical samples in a query.
- [x] None of the above

Is the submission congruent with all relevant MLPerf rules?
- [x] Yes
- [ ] No

For each SUT, does the submission accurately reflect the real-world
performance of the SUT?
- [x] Yes
- [ ] No"""

def get_checklist_0_5(checklist_template=checklist_template_0_5,
        name='Anton Lokhmotov', email='anton@dividiti.com',
        system='rpi4-tflite-v1.15', system_name='Raspberry Pi 4 (rpi4)', revision='61220457de',
        division='closed', category='available', task='image-classification', benchmark='mobilenet', scenario='singlestream',
        performance_sample_count=1024, performance_sample_count_met=True,
        accuracy_pc=12.345, accuracy_met=True, numerics='fp32'):
    def tick(var): return "x" if var else " "
    print("=" * 100)
    print(system)
    print("=" * 100)
    revision_other = revision not in [ '61220457de', '5684c11e39', '55c0ea4e77', 'd31c18fbd9', '1d0e06e54a' ]
    benchmark_other = benchmark not in [ 'mobilenet', 'resnet']
    if benchmark=='mobilenet':
        accuracy_met = accuracy_pc >= 71.676*0.98
    elif benchmark=='resnet':
        accuracy_met = accuracy_pc >= 76.456*0.99
    else:
        accuracy_met = accuracy_met and accuracy_pc > 0
    checklist = checklist_template % {
        "name" : name,
        "email" : email,
        "system_name": system_name,
        # Division.
        "closed" : tick(division=='closed'),
        "open" : tick(division=='open'),
        # Category.
        "category_available" : tick(category.lower()=='available'),
        "category_preview" : tick(category.lower()=='preview'),
        "category_rdi" : tick(category.lower()=='rdi'),
        # Benchmark.
        "benchmark_mobilenet": tick(benchmark=='mobilenet'),
        "benchmark_resnet": tick(benchmark=='resnet'),
        "benchmark_other": tick(benchmark_other),
        "benchmark_other_specify": benchmark if benchmark_other else '',
        # Table.
        "system" : system,
        "benchmark" : benchmark,
        "query_count": 50000 if task=='image-classification' else 5000,
        "accuracy_pc" : "%.3f" % accuracy_pc,
        # Scenario.
        "scenario_singlestream": tick(scenario=='singlestream'),
        "scenario_multistream": tick(scenario=='multistream'),
        "scenario_server": tick(scenario=='server'),
        "scenario_offline": tick(scenario=='offline'),
        # Accuracy.
        "mobilenet_accuracy_met" : tick(benchmark=='mobilenet' and accuracy_met),
        "resnet_accuracy_met" : tick(benchmark=='resnet' and accuracy_met),
        "accuracy_not_met" : tick(not accuracy_met),
        # "How many samples are loaded into the QSL in performance mode?"
        "performance_sample_count": performance_sample_count,
        "performance_sample_count_1024": tick(performance_sample_count==1024),
        "performance_sample_count_256": tick(performance_sample_count==256),
        "performance_sample_count_64": tick(performance_sample_count==64),
        "performance_sample_count_not_met": tick(not performance_sample_count_met), # TODO
        # LoadGen revision.
        "revision_61220457de": tick(revision=='61220457de'),
        "revision_5684c11e39": tick(revision=='5684c11e39'),
        "revision_55c0ea4e77": tick(revision=='55c0ea4e77'),
        "revision_d31c18fbd9": tick(revision=='d31c18fbd9'),
        "revision_1d0e06e54a": tick(revision=='1d0e06e54a'),
        "revision_other":  tick(revision_other),
        "revision_other_specify": revision if revision_other else '',
        # Numerics.
        "numerics_uint8": tick(numerics=='uint8'),
        "numerics_fp32": tick(numerics=='fp32'),
    }
    print(checklist)
    print("-" * 100)

    return checklist

# null = get_checklist_0_5(system='rpi4-armnn-v19.08-neon', system_name='Raspberry Pi 4 (rpi4)', benchmark='mobilenet', accuracy_pc=70.241, numerics='uint8')
# null = get_checklist_0_5(system='hikey960-tflite-v1.15', system_name='Linaro HiKey 960 (hikey960)', benchmark='resnet', accuracy_pc=75.692, revision='deadbeef')
# null = get_checklist_0_5(system='velociti-tensorflow-v1.14-cpu', name='Anton Lokhmotov; Emanuele Vitali', email='anton@dividiti.com; emanuele.vitali@polimi.it', system_name='HP Z640 G1X62EA workstation (velociti)', division='open', category='RDI', benchmark='ssd-mobilenet-fpn')


# <a id="check"></a>
# ## Check the experimental data

# In[ ]:


#
# Image Classification - Closed (MobileNet, ResNet).
#
repos_image_classification_closed = [
    # firefly
    'mlperf.closed.image-classification.firefly.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/18
    'mlperf.closed.image-classification.firefly.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/21
    'mlperf.closed.image-classification.firefly.armnn-v19.08.opencl', #https://github.com/mlperf/submissions_inference_0_5/pull/22
    # hikey960
    'mlperf.closed.image-classification.hikey960.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/23
    'mlperf.closed.image-classification.hikey960.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/24
    'mlperf.closed.image-classification.hikey960.armnn-v19.08.opencl', # https://github.com/mlperf/submissions_inference_0_5/pull/25
    # rpi4
    'mlperf.closed.image-classification.rpi4.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/26/
    'mlperf.closed.image-classification.rpi4.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/30
    # mate10pro
    'mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/32
    'mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl', # https://github.com/mlperf/submissions_inference_0_5/pull/35
]

repos_image_classification_closed_audit = [
    'mlperf.closed.image-classification.firefly.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/234
    'mlperf.closed.image-classification.hikey960.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/236
    'mlperf.closed.image-classification.rpi4.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/238
    #'mlperf.closed.image-classification.mate10pro.audit',
]

#
# Image Classification - Open (MobileNets-v1,v2).
#
repos_image_classification_open = [
    # firefly
    'mlperf.open.image-classification.firefly.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/39
    'mlperf.open.image-classification.firefly.tflite-v1.15.mobilenet-v1-quantized', # https://github.com/mlperf/submissions_inference_0_5/pull/127
    'mlperf.open.image-classification.firefly.armnn-v19.08.opencl', # https://github.com/mlperf/submissions_inference_0_5/pull/40
    'mlperf.open.image-classification.firefly.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/120
    # hikey960
    'mlperf.open.image-classification.hikey960.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/37
    'mlperf.open.image-classification.hikey960.tflite-v1.15.mobilenet-v1-quantized', # https://github.com/mlperf/submissions_inference_0_5/pull/128
    'mlperf.open.image-classification.hikey960.armnn-v19.08.opencl', # https://github.com/mlperf/submissions_inference_0_5/pull/38
    'mlperf.open.image-classification.hikey960.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/121
    # rpi4
    'mlperf.open.image-classification.rpi4.tflite-v1.15', # https://github.com/mlperf/submissions_inference_0_5/pull/122
    'mlperf.open.image-classification.rpi4.tflite-v1.15.mobilenet-v1-quantized', # https://github.com/mlperf/submissions_inference_0_5/pull/129
    'mlperf.open.image-classification.rpi4.armnn-v19.08.neon', # https://github.com/mlperf/submissions_inference_0_5/pull/123
    # mate10pro
    'mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet', # https://github.com/mlperf/submissions_inference_0_5/pull/130
    'mlperf.open.image-classification.mate10pro.tflite-v1.13.mobilenet-v1-quantized', # https://github.com/mlperf/submissions_inference_0_5/pull/135
]

repos_image_classification_open_audit = [
    'mlperf.open.image-classification.firefly.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/255
    'mlperf.open.image-classification.hikey960.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/257
    'mlperf.open.image-classification.rpi4.audit', # https://github.com/mlperf/submissions_inference_0_5/pull/258
    #'mlperf.open.image-classification.mate10pro.audit',
]

#
# Object Detection - Open (TensorFlow Model Zoo + YOLO-v3)
#
repos_object_detection_open = [
    # velociti
    'mlperf.open.object-detection.velociti', # https://www.dropbox.com/s/wiea3a8zf077jsv/mlperf.open.object-detection.velociti.zip
]


# In[ ]:


# repos_for_testing = [
#     'mlperf.closed.image-classification.mate10pro.tflite-v1.13.mobilenet.BAD_LOADGEN',
#     'mlperf.closed.image-classification.mate10pro.armnn-v19.08.opencl.BAD_RESNET',
#     'mlperf.closed.image-classification.mate10pro.armnn-v19.08.neon.BAD_RESNET',
#     'mlperf-inference-vision-experiments-count5'
# ]


# In[ ]:


# #!ck recache repo
# for repo_uoa in repos:
#     print("=" * 100)
#     print(repo_uoa)
#     print("=" * 100)
#     !ck list $repo_uoa:experiment:* | sort
#     print("-" * 100)
#     print("")


# In[ ]:

upstream_path=os.environ.get('CK_ENV_MLPERF_INFERENCE','')
vlatest_path=os.environ.get('CK_ENV_MLPERF_INFERENCE_VLATEST')

# In[ ]:

root_dir=os.environ.get('CK_MLPERF_SUBMISSION_ROOT','')

def check_experimental_results(repo_uoa, module_uoa='experiment', tags='mlperf', extra_tags='',
                               submitter='dividiti', submitter_desc='dividiti', path=None,
                               compliance=False, version='v0.7', infer_offline_from_singlestream=False):
    if not os.path.exists(root_dir): os.mkdir(root_dir)
    print("Storing results under '%s'" % root_dir)

    if extra_tags:
        tags += ',' + extra_tags
    r = ck.access({'action':'search', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'tags':tags})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)
    experiments = r['lst']
    print("Found {} {} entries in repository '{}'".format(len(experiments), module_uoa, repo_uoa))

    for experiment in experiments:
        data_uoa = experiment['data_uoa']
        repo_uoa = experiment['repo_uoa']
        r = ck.access({'action':'list_points', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
        if r['return']>0:
            print('Error: %s' % r['error'])
            exit(1)

        experiment_tags     = r['dict']['tags']
        experiment_points   = r['points']
        experiment_path     = r['path']

        # Load pipeline to determine the original program_name
        load_pipeline_adict = { 'action':           'load_pipeline',
                                'repo_uoa':         repo_uoa,
                                'module_uoa':       module_uoa,
                                'data_uoa':         data_uoa,
        }
        r=ck.access( load_pipeline_adict )
        if r['return']>0:
            print('Error: %s' % r['error'])
            exit(1)

        pipeline = r['pipeline']
        program_name = pipeline['choices']['data_uoa']

        print("*" * 100)

        division=task=platform=library=inference_engine=backend=benchmark=scenario=mode=preprocessing=test=notes = ''

        for atag in experiment_tags:
            if '.' in atag:     # Expected format: attribute1.value1 , attribute2.value2 , etc - in any order
                # Example:   "division.open", "submitter.dividiti", "task.image-classification", "platform.xavier",
                # "inference_engine.tflite", "inference_engine_version.v2.1.1", "inference_engine_backend.dummy",
                # "workload.mobilenet-v2-1.4-224", "scenario.singlestream", "mode.performance"
                (attribute, value) = atag.split('.', 1)     # protection from dotted version notation!
                if attribute == 'division':
                    division = value
                elif attribute == 'task':
                    task = value
                elif attribute == 'platform':
                    platform = value
                elif attribute == 'inference_engine':
                    inference_engine = value
                elif attribute == 'inference_engine_version':
                    inference_engine_version = value
                elif attribute == 'inference_engine_backend':
                    backend = value if value!='dummy' else ''
                elif attribute == 'workload':   # actually, the model!
                    benchmark = value
                elif attribute == 'scenario':
                    scenario = value
                elif attribute == 'mode':
                    mode = value
                elif attribute == 'compliance':
                    test = value

        # Skip performance and accuracy experiments when doing a compliance pass.
        if compliance and test == '':
            continue
        # Skip compliance experiments when not doing a compliance pass.
        if not compliance and test != '':
            continue

        if division and task and platform and inference_engine and benchmark and scenario and mode:
            library = inference_engine + (('-' + inference_engine_version) if inference_engine_version else '')

        elif 'velociti' in experiment_tags:
            # Expected format: [ "mlperf", "open", "object-detection", "velociti", "cpu", "rcnn-inception-resnet-v2-lowproposals", "singlestream", "accuracy" ]
            (_, division, task, platform, backend, benchmark, scenario, mode) = experiment_tags
            if task == 'object-detection':
                library = 'tensorflow-v1.14'
            else:
                library = 'tensorrt-v6.0'
                backend = ''
                notes = '======= DEMO ======='
        elif 'accuracy' in experiment_tags:
            # FIXME: With the benefit of hindsight, [ ..., "armnn-v19.08", "neon", ... ] should have come
            # as one tag "armnn-v19.08-neon", since we join them in this notebook anyway.
            if 'neon' in experiment_tags or 'opencl' in experiment_tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "accuracy", "using-opencv" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode, preprocessing) = experiment_tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "accuracy", "using-opencv" ]
                (_, division, task, platform, library, benchmark, scenario, mode, preprocessing) = experiment_tags
        elif 'performance' in experiment_tags:
            if 'neon' in experiment_tags or 'opencl' in experiment_tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "performance" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode) = experiment_tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "performance" ]
                (_, division, task, platform, library, benchmark, scenario, mode) = experiment_tags
        elif 'audit' in experiment_tags: # As accuracy but with the test name instead of the preprocessing method.
            if 'neon' in experiment_tags or 'opencl' in experiment_tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "audit", "TEST03" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode, test) = experiment_tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "audit", "TEST03" ]
                (_, division, task, platform, library, benchmark, scenario, mode, test) = experiment_tags
        else:
            raise Exception("Expected 'accuracy' or 'performance' or 'audit' in experiment_tags!")

        global any_closed_results
        if division.lower() == 'closed':
            any_closed_results = True

        global any_open_results
        if division.lower() == 'open':
            any_open_results = True

        organization = submitter

        if not inference_engine:
            (inference_engine, inference_engine_version) = library.split('-')

        if backend != '':
            system = platform+'-'+library+'-'+backend
        else:
            system = platform+'-'+library
        division_system = division+'-'+system

        program_and_model_combination = program_name+'-'+benchmark
        #
        # Directory structure according to the Inference section of the General MLPerf Submission Rules:
        # https://github.com/mlperf/policies/blob/master/submission_rules.adoc#552-inference
        #
        # <division>/
        #   <organization>/
        #
        division_dir = os.path.join(root_dir, division)
        if not os.path.exists(division_dir): os.mkdir(division_dir)
        organization_dir = os.path.join(division_dir, organization)
        if not os.path.exists(organization_dir): os.mkdir(organization_dir)

        #
        #     "systems"/
        #       <system_desc_id>.json
        #
        systems_dir = os.path.join(organization_dir, 'systems')
        if not os.path.exists(systems_dir): os.mkdir(systems_dir)
        system_json_name = '%s.json' % system
        system_json_path = os.path.join(systems_dir, system_json_name)

        system_json = dump_system_description_dictionary(system_json_path, submitter_desc, division, platform, inference_engine, inference_engine_version, backend)
        print('%s' % systems_dir)
        print('  |_ %s [%s]' % (system_json_name, division_system))

        #
        #     "code"/
        #       <benchmark_name_per_reference>/
        #         <implementation_id>/
        #           <Code interface with loadgen and other arbitrary stuff>
        #
        code_dir = os.path.join(organization_dir, 'code')
        if not os.path.exists(code_dir): os.mkdir(code_dir)
        # FIXME: For now, not always "per reference".
        benchmark_dir = os.path.join(code_dir, benchmark)
        if not os.path.exists(benchmark_dir): os.mkdir(benchmark_dir)
        implementation_dir = os.path.join(benchmark_dir, program_name)
        if not os.path.exists(implementation_dir): os.mkdir(implementation_dir)
        print('%s' % code_dir)

        # Create 'README.md'.
        implementation_readme_name = 'README.md'
        implementation_readme_path = os.path.join(implementation_dir, implementation_readme_name)
        if program_name in [ 'openvino-loadgen-v0.7-drop' ]:
            program_path = get_program_path(program_name)
            program_readme_name = 'README.setup.md'
            program_readme_path = os.path.join(program_path, program_readme_name)
            copy2(program_readme_path, implementation_readme_path)
            print('  |_ %s [from %s]' % (implementation_readme_name, program_readme_path))
        elif program_name in [ 'image-classification-tflite-loadgen', 'image-classification-armnn-tflite-loadgen', 'object-detection-tflite-loadgen' ]:
            program_path = get_program_path(program_name)
            program_readme_name = 'README.md'
            program_readme_path = os.path.join(program_path, program_readme_name)
            copy2(program_readme_path, implementation_readme_path)
            print('  |_ %s [from %s]' % (implementation_readme_name, program_readme_path))
        else: # v0.5
#           pprint(implementation_readmes)
            implementation_readme = implementation_readmes.get(program_name, '')
            with open(implementation_readme_path, 'w') as implementation_readme_file:
                implementation_readme_file.writelines(implementation_readme)
            if implementation_readme == '':
                print('  |_ %s [EMPTY]' % implementation_readme_name)
            # raise
            else:
                print('  |_ %s' % implementation_readme_name)

        #
        #     "measurements"/
        #       <system_desc_id>/
        #         <benchmark>/
        #           <scenario>/
        #             <system_desc_id>_<implementation_id>.json
        #             README.md
        #             user.conf
        #             mlperf.conf
        #             calibration_process.adoc (?)
        #             submission_checklist.txt
        #
        measurements_dir = os.path.join(organization_dir, 'measurements')
        if not os.path.exists(measurements_dir): os.mkdir(measurements_dir)
        system_dir = os.path.join(measurements_dir, system)
        if not os.path.exists(system_dir): os.mkdir(system_dir)
        benchmark_dir = os.path.join(system_dir, benchmark)
        if not os.path.exists(benchmark_dir): os.mkdir(benchmark_dir)
        mscenario_dir = os.path.join(benchmark_dir, scenario)
        if not os.path.exists(mscenario_dir): os.mkdir(mscenario_dir)
        print(mscenario_dir)

        # Create '<system_desc_id>_<implementation_id>.json'.
        system_implementation_json_name = system+'_'+program_name+'.json'
        system_implementation_json_path = os.path.join(mscenario_dir, system_implementation_json_name)

        implementation_benchmark_json = dump_implementation_dictionary(system_implementation_json_path, pipeline['dependencies']['weights'], inference_engine, program_name, benchmark)
        print('  |_ %s [for %s]' % (system_implementation_json_name, program_and_model_combination))

        if version == 'v0.5':
            # Create 'README.md' based on the division and task (basically, mentions a division- and task-specific script).
            measurements_readme_name = 'README.md'
            measurements_readme_path = os.path.join(mscenario_dir, measurements_readme_name)
            measurements_readme = measurements_readmes.get(division+'-'+task, '')
            if measurements_readme != '':
                with open(measurements_readme_path, 'w') as measurements_readme_file:
                    measurements_readme_file.writelines(measurements_readme)
                print('  |_ %s [for %s %s]' % (measurements_readme_name, division, task))
            else:
                raise Exception("Invalid measurements README!")
        else:
            if program_name == 'openvino-loadgen-v0.7-drop':
                program_readme_name = 'README.{}-{}.md'.format(benchmark, scenario)
            elif program_name in [ 'image-classification-tflite-loadgen', 'image-classification-armnn-tflite-loadgen', 'object-detection-tflite-loadgen' ]:
                program_readme_name = 'README.{}.md'.format(scenario)
            elif program_name in [ 'image-classification-tensorrt-loadgen-py', 'object-detection-tensorrt-loadgen-py' ]:
                program_readme_name = 'README.{}.md'.format(benchmark)

            program_path = get_program_path(program_name)
            program_readme_path = os.path.join(program_path, program_readme_name)
            measurements_readme_name = 'README.md'
            measurements_readme_path = os.path.join(mscenario_dir, measurements_readme_name)
            copy2(program_readme_path, measurements_readme_path)
            print('  |_ %s [from %s]' % (measurements_readme_name, program_readme_path))

        # Create 'NOTES.txt'.
        measurements_notes_name = 'NOTES.txt'
        measurements_notes_path = os.path.join(mscenario_dir, measurements_notes_name)
        measurements_notes = notes
        if measurements_notes != '':
            with open(measurements_notes_path, 'w') as measurements_notes_file:
                measurements_notes_file.writelines(measurements_notes)
            print('  |_ %s [for %s %s]' % (measurements_notes_name, division, task))


        # With newer programs instead of per-program configs we have recorded per-run configs, which will be dumped later elsewhere
        if version == 'v0.5':

            # Try to find environment for 'user.conf'.
            if program_name.endswith('-loadgen'):
                program_config_tag = program_name[:-len('-loadgen')]
            else:
                program_config_tag = program_name
            loadgen_config_tags='loadgen,config,'+program_config_tag    # FIXME: needs to be fixed on the soft: entry side
            lgc = ck.access({'action':'search', 'module_uoa':'env', 'tags':loadgen_config_tags})
            if lgc['return']>0:
                print('Error: %s' % lgc['error'])
                exit(1)
            envs = lgc['lst']
            if len(envs) > 1:
               # Found several environments.
               print('Error: More than one environment found with tags=\'%s\'' % loadgen_config_tags)
               exit(1)
            elif len(envs) == 1:
                # Found exactly one environment.
                lgc = ck.access({'action':'load', 'module_uoa':'env', 'data_uoa':envs[0]['data_uoa']})
                if lgc['return']>0:
                    print('Error: %s' % lgc['error'])
                    exit(1)
                # CK_ENV_LOADGEN_CONFIG=/home/anton/CK_REPOS/ck-mlperf/soft/config.loadgen/image-classification-armnn-tflite-loadgen-conf
                # CK_ENV_LOADGEN_CONFIG_FILE=/home/anton/CK_REPOS/ck-mlperf/soft/config.loadgen/image-classification-armnn-tflite-loadgen-conf/user.conf
                user_conf_path=lgc['dict']['env']['CK_ENV_LOADGEN_CONFIG_FILE']
                user_conf_name=user_conf_path[len(lgc['dict']['env']['CK_ENV_LOADGEN_CONFIG'])+1:]
            elif len(envs) == 0:
                # Not found any environments: copy 'user.conf' from implementation source.
                user_conf_name = 'user.conf'
                implementation_path = get_program_path(program_name)
                if not implementation_path:
                    raise Exception("Invalid implementation path!")
                user_conf_path = os.path.join(implementation_path, user_conf_name)
            copy2(user_conf_path, mscenario_dir)
            print('  |_ %s [from %s]' % (user_conf_name, user_conf_path))

            # Copy 'mlperf.conf' from MLPerf Inference source.
            mlperf_conf_name = 'mlperf.conf'
            mlperf_conf_path = os.path.join(mscenario_dir, mlperf_conf_name)
            if program_name in [ 'image-classification-tflite-loadgen', 'image-classification-armnn-tflite-loadgen' ]:
                # Write a snapshot from https://github.com/dividiti/inference/blob/61220457dec221ed1984c62bd9d382698bd71bc6/v0.5/mlperf.conf
                with open(mlperf_conf_path, 'w') as mlperf_conf_file:
                    mlperf_conf_file.writelines(mlperf_conf_6122045)
                print('  |_ %s [from %s]' % (mlperf_conf_name, 'github.com/mlperf/inference@6122045'))
            else:
                upstream_mlperf_conf_path = os.path.join(upstream_path, 'v0.5', 'mlperf.conf')
                copy2(upstream_mlperf_conf_path, mlperf_conf_path)
                print('  |_ %s [from %s]' % (mlperf_conf_name, upstream_mlperf_conf_path))

        # Write submission_checklist.txt into the same directory later, once accuracy.txt is parsed.

        #
        #     "results"/
        #       <system_desc_id>/
        #         <benchmark>/
        #           <scenario>/
        #             performance/
        #               run_x/ # 1 run for single stream and offline, 5 otherwise
        #                 mlperf_log_summary.txt
        #                 mlperf_log_detail.txt
        #                 mlperf_log_trace.json
        #             accuracy/
        #               mlperf_log_accuracy.json
        #         submission_checker_log.txt
        #
        results_dir = os.path.join(organization_dir, 'results')
        if not os.path.exists(results_dir): os.mkdir(results_dir)
        system_dir = os.path.join(results_dir, system)
        if not os.path.exists(system_dir): os.mkdir(system_dir)
        benchmark_dir = os.path.join(system_dir, benchmark)
        if not os.path.exists(benchmark_dir): os.mkdir(benchmark_dir)
        scenario_dir = os.path.join(benchmark_dir, scenario)
        if not os.path.exists(scenario_dir): os.mkdir(scenario_dir)
        mode_dir = os.path.join(scenario_dir, mode)
        if not os.path.exists(mode_dir): os.mkdir(mode_dir)
        # Create a compliance directory structure.
        if compliance:
            # Deal with a subset of tests.
#             if test not in [ 'TEST03' ]: # [ 'TEST01', 'TEST03', 'TEST04-A', 'TEST04-B', 'TEST05' ]:
#                 continue
            # Save the accuracy and performance dirs for the corresponding submission experiment.
            accuracy_dir = os.path.join(scenario_dir, 'accuracy')
            performance_dir = os.path.join(scenario_dir, 'performance', 'run_1')
            # Use the mode expected for each test.
            mode = 'performance' if test != 'TEST03' else 'submission'
            # Create a similar directory structure to results_dir, with another level, test_dir,
            # between scenario_dir and mode_dir.
            if version == 'v0.5':
                audit_dir = os.path.join(organization_dir, 'audit')
                if not os.path.exists(audit_dir): os.mkdir(audit_dir)
                system_dir = os.path.join(audit_dir, system)
            else: # v0.7+
                compliance_dir = os.path.join(organization_dir, 'compliance')
                if not os.path.exists(compliance_dir): os.mkdir(compliance_dir)
                system_dir = os.path.join(compliance_dir, system)
            if not os.path.exists(system_dir): os.mkdir(system_dir)
            benchmark_dir = os.path.join(system_dir, benchmark)
            if not os.path.exists(benchmark_dir): os.mkdir(benchmark_dir)
            scenario_dir = os.path.join(benchmark_dir, scenario)
            if not os.path.exists(scenario_dir): os.mkdir(scenario_dir)
            test_dir = os.path.join(scenario_dir, test)
            if not os.path.exists(test_dir): os.mkdir(test_dir)
            mode_dir = os.path.join(test_dir, mode)
            if not os.path.exists(mode_dir): os.mkdir(mode_dir)
        print(mode_dir)

        run_idx = 0
        # For each experiment point (can be more than one if manually combining data from separate runs).
        for point in experiment_points:
            point_file_path = os.path.join(experiment_path, 'ckp-%s.0001.json' % point)
            with open(point_file_path) as point_file:
                point_data_raw = json.load(point_file)
            characteristics_list = point_data_raw['characteristics_list']
            # For each repetition (can be more than one for server).
            for characteristics in characteristics_list:
                run_idx += 1
                # Set the leaf directory.
                if mode == 'performance':
                    run_dir = os.path.join(mode_dir, 'run_%d' % run_idx)
                    if not os.path.exists(run_dir): os.mkdir(run_dir)
                    last_dir = run_dir
                    # Performance notes. Should ideally go inside the run_x dir, but the checker complains.
                    if 'velociti' in experiment_tags and 'tensorrt' in experiment_tags:
                        num_streams = point_data_raw['choices']['env'].get('CK_LOADGEN_MULTISTREAMNESS', '')
                        if num_streams == '': num_streams = '?'
                        performance_notes = 'uid={}: {} streams'.format(point, num_streams)
                        performance_notes_name = run_dir + '.txt'
                        performance_notes_path = os.path.join(mode_dir, performance_notes_name)
                        with open(performance_notes_path, 'w') as performance_notes_file:
                            performance_notes_file.writelines(performance_notes)
                            print('  |_ %s' % performance_notes_name)
                else:
                    last_dir = mode_dir
                print(last_dir)

                mlperf_conf = characteristics['run'].get('mlperf_conf',{})
                for config_name in mlperf_conf.keys():
                    # FIXME: Per-datapoint user configs belong here, but this must be confirmed!
                    # https://github.com/mlperf/policies/issues/57
                    #config_dir = {'mlperf.conf': mscenario_dir, 'user.conf': last_dir, 'audit.config': last_dir}[config_name]

                    # For now, still write under measurements/ according to the current rules.
                    config_dir = {'mlperf.conf': mscenario_dir, 'user.conf': mscenario_dir}.get(config_name)
                    if not config_dir: continue

                    full_config_path = config_path = os.path.join(config_dir, config_name)
                    write_config = True
                    if os.path.exists(full_config_path):
                        with open(full_config_path, 'r') as existing_config_fd:
                            existing_config_lines = existing_config_fd.readlines()

                        if set(existing_config_lines) == set(mlperf_conf[config_name]):
                            print("Found an identical '{}' file, skipping it".format(full_config_path))
                            write_config = False
                        else:
                            print("Warning: Found an existing '{}' file with different contents:\n-- old: {}\n-- new: {}".format(full_config_path, existing_config_lines, mlperf_conf[config_name]))
                            if set(existing_config_lines) > set(mlperf_conf[config_name]):
                                print("The existing config fully includes the new candidate, keeping the existing one")
                                write_config = False
                            elif set(existing_config_lines) < set(mlperf_conf[config_name]):
                                print("The existing config is fully contained in the new candidate, overwriting the existing one")
                            else:
                                #raise Exception("Probably a conflict, please investigate!")
                                write_config = False

                    if write_config:
                        with open(config_path, 'w') as new_config_fd:
                            new_config_fd.writelines(mlperf_conf[config_name])
                        print('  |_ {}'.format(config_name))

                # Dump files in the leaf directory.
                mlperf_log = characteristics['run'].get('mlperf_log',{})
                # Summary file (with errors and warnings in accuracy mode, with statistics in performance mode).
                summary = mlperf_log.get('summary','')
                # Check that the summary corresponds to a VALID run, skip otherwise.
                parsed_summary = {}
                for line in summary:
                  pair = line.strip().split(': ', 1)
                  if len(pair)==2:
                    parsed_summary[ pair[0].strip() ] = pair[1].strip()
                if mode == 'performance' and parsed_summary['Result is'] != 'VALID' and not compliance:
                  run_idx -= 1
                  continue
                summary_txt_name = 'mlperf_log_summary.txt'
                summary_txt_path = os.path.join(last_dir, summary_txt_name)
                with open(summary_txt_path, 'w') as summary_txt_file:
                    summary_txt_file.writelines(summary)
                    print('  |_ %s' % summary_txt_name)
                # Detail file (with settings).
                detail_txt_name = 'mlperf_log_detail.txt'
                detail_txt_path = os.path.join(last_dir, detail_txt_name)
                detail = mlperf_log.get('detail','')
                with open(detail_txt_path, 'w') as detail_txt_file:
                    detail_txt_file.writelines(detail)
                    print('  |_ %s' % detail_txt_name)
                # Accuracy file (with accuracy dictionary).
                if mode == 'accuracy':
                    accuracy_json_name = 'mlperf_log_accuracy.json'
                    accuracy_json_path = os.path.join(last_dir, accuracy_json_name)
                    with open(accuracy_json_path, 'w') as accuracy_json_file:
                        json.dump(mlperf_log.get('accuracy',{}), accuracy_json_file, indent=2)
                        print('  |_ %s' % accuracy_json_name)
                # Accuracy file (with accuracy dictionary).
                if compliance and test == 'TEST01':
                    accuracy_json_name = 'mlperf_log_accuracy.json'
                    # The script for truncating accuracy logs expects to find the log under 'TEST01/accuracy'.
                    fake_accuracy_dir = os.path.join(test_dir, 'accuracy')
                    if not os.path.exists(fake_accuracy_dir): os.mkdir(fake_accuracy_dir)
                    accuracy_json_path = os.path.join(fake_accuracy_dir, accuracy_json_name)
                    with open(accuracy_json_path, 'w') as accuracy_json_file:
                        json.dump(mlperf_log.get('accuracy',{}), accuracy_json_file, indent=2)
                        print('  |_ %s' % accuracy_json_name)
                # Do what's required by NVIDIA's compliance tests.
                if compliance:
                    if version == 'v0.5':
                        test_path = os.path.join(upstream_path, 'v0.5', 'audit', 'nvidia', test)
                    else: # v0.7+
                        test_path = os.path.join(upstream_path, 'compliance', 'nvidia', test)
                    if 'TEST01' in experiment_tags or test == 'TEST01':
                        # Verify that the accuracy (partially) dumped for the test matches that for the submision.
                        if version == 'v0.5':
                            submission_accuracy_flag = '-a'
                            test_accuracy_flag = '-p'
                        else: # v0.7+
                            submission_accuracy_flag = '-r'
                            test_accuracy_flag = '-t'
                        verify_accuracy_py = os.path.join(test_path, 'verify_accuracy.py')
                        submission_accuracy_json_path = os.path.join(accuracy_dir, accuracy_json_name)
                        print("submission_accuracy_json_path={}".format(submission_accuracy_json_path))
                        print("accuracy_json_path={}".format(accuracy_json_path))
                        verify_accuracy_txt = subprocess.getoutput('python3 {} {} {} {} {}' \
                                              .format(verify_accuracy_py, submission_accuracy_flag, submission_accuracy_json_path, test_accuracy_flag, accuracy_json_path))
                        verify_accuracy_txt_name = 'verify_accuracy.txt'
                        verify_accuracy_txt_path = os.path.join(test_dir, verify_accuracy_txt_name)
                        with open(verify_accuracy_txt_path, 'w') as verify_accuracy_txt_file:
                            verify_accuracy_txt_file.write(verify_accuracy_txt)
                            print('%s' % test_dir)
                            print('  |_ %s' % verify_accuracy_txt_name)
                    if test in [ 'TEST01', 'TEST03', 'TEST05' ]:
                        # Verify that the performance for the test matches that for the submission.
                        verify_performance_py = os.path.join(test_path, 'verify_performance.py')
                        submission_summary_txt_path = os.path.join(performance_dir, summary_txt_name)
                        verify_performance_txt = subprocess.getoutput('python3 {} -r {} -t {}'.format(verify_performance_py, submission_summary_txt_path, summary_txt_path))
                        verify_performance_txt += os.linesep
                        verify_performance_txt_name = 'verify_performance.txt'
                        verify_performance_txt_path = os.path.join(test_dir, verify_performance_txt_name)
                        with open(verify_performance_txt_path, 'w') as verify_performance_txt_file:
                            verify_performance_txt_file.write(verify_performance_txt)
                            print('%s' % test_dir)
                            print('  |_ %s' % verify_performance_txt_name)
                    if test in [ 'TEST04-A', 'TEST04-B' ]:
                        test04a_summary_txt_path = os.path.join(scenario_dir, 'TEST04-A', 'performance', 'run_1', summary_txt_name)
                        test04b_summary_txt_path = os.path.join(scenario_dir, 'TEST04-B', 'performance', 'run_1', summary_txt_name)
                        if os.path.exists(test04a_summary_txt_path) and os.path.exists(test04b_summary_txt_path):
                            # If both tests have been processed, verify that their performance matches.
                            if version == 'v0.5':
                                verify_performance_py = os.path.join(upstream_path, 'v0.5', 'audit', 'nvidia', 'TEST04-A', 'verify_test4_performance.py')
                            else:
                                verify_performance_py = os.path.join(upstream_path, 'compliance', 'nvidia', 'TEST04-A', 'verify_test4_performance.py')
                            #print("python3 {} -u {} -s {}".format(verify_performance_py, test04a_summary_txt_path, test04b_summary_txt_path))
                            verify_performance_txt = subprocess.getoutput('python3 {} -u {} -s {}'.format(verify_performance_py, test04a_summary_txt_path, test04b_summary_txt_path))
                            #print(verify_performance_txt)
                            verify_performance_txt_name = 'verify_performance.txt'
                            verify_performance_txt_path = os.path.join(scenario_dir, 'TEST04-A', verify_performance_txt_name)
                            with open(verify_performance_txt_path, 'w') as verify_performance_txt_file:
                                verify_performance_txt_file.write(verify_performance_txt)
                                print('%s' % os.path.join(scenario_dir, 'TEST04-A'))
                                print('  |_ %s' % verify_performance_txt_name)
                        else:
                            # Need both A/B tests to be processed. Wait for the other one.
                            continue
                # Generate accuracy.txt.
                if mode == 'accuracy' or mode == 'submission' or (compliance and test == 'TEST01'):
                    accuracy_txt_name = 'accuracy.txt'
                    if compliance and test == 'TEST01':
                        accuracy_txt_path = os.path.join(fake_accuracy_dir, accuracy_txt_name)
                    else:
                        accuracy_txt_path = os.path.join(last_dir, accuracy_txt_name)
                    if task == 'image-classification':
                        accuracy_imagenet_py = os.path.join(vlatest_path, 'classification_and_detection', 'tools', 'accuracy-imagenet.py')
                        accuracy_cmd = 'python3 {} --imagenet-val-file {} --mlperf-accuracy-file {}'.format(accuracy_imagenet_py, imagenet_val_file, accuracy_json_path)
                        accuracy_txt = subprocess.getoutput(accuracy_cmd)
                        # The last (and only) line is e.g. "accuracy=76.442%, good=38221, total=50000".
                        accuracy_line = accuracy_txt.split('\n')[-1]
                        match = re.match('accuracy=(.+)%, good=(\d+), total=(\d+)', accuracy_line)
                        accuracy_pc = float(match.group(1))
                    elif task == 'object-detection':
                        accuracy_coco_py = os.path.join(vlatest_path, 'classification_and_detection', 'tools', 'accuracy-coco.py')
                        use_inv_map = bool(int(pipeline['dependencies']['weights']['cus']['install_env'].get('ML_MODEL_USE_INV_MAP','0')))
                        use_inv_map_flag = '--use-inv-map' if use_inv_map else ''
                        accuracy_cmd = 'python3 {} --coco-dir {} --mlperf-accuracy-file {} {}'.format(accuracy_coco_py, coco_dir, accuracy_json_path, use_inv_map_flag)
                        accuracy_txt = subprocess.getoutput( accuracy_cmd )
                        # The last line is e.g. "mAP=13.323%".
                        accuracy_line = accuracy_txt.split('\n')[-1]
                        match = re.match('mAP\=([\d\.]+)\%', accuracy_line)
                        if match:
                            accuracy_pc = float(match.group(1))
                        else:
                            raise Exception("Could not parse accuracy from: '{}' by running the command:\n\t{}".format(accuracy_txt, accuracy_cmd))
                    else:
                        raise Exception("Invalid task '%s'!" % task)
                    with open(accuracy_txt_path, 'w') as accuracy_txt_file:
                        print('  |_ %s [%.3f%% parsed from "%s"]' % (accuracy_txt_name, accuracy_pc, accuracy_line))
                        # Append a new line for the truncate script to work as expected.
                        accuracy_txt += os.linesep
                        accuracy_txt_file.write(accuracy_txt)

                # Generate submission_checklist.txt for each system, benchmark and scenario under "measurements/".
                if mode == 'accuracy' and not compliance:
                    checklist_name = 'submission_checklist.txt'
                    checklist_path = os.path.join(measurements_dir, system, benchmark, scenario, checklist_name)
                    # Write the checklist.
                    if submitter.lower() == 'dividiti':
                        checklist = get_checklist_0_7(name='Anton Lokhmotov', email='anton@dividiti.com',
                                                      system_name=system_json['system_name'],
                                                      numerics=implementation_benchmark_json['weight_data_types'])

                    elif submitter.lower() == 'dellemc':
                        checklist = get_checklist_0_7(name='Vilmara Sanchez', email='Vilmara_Sanchez@dellteam.com',
                                                      system_name=system_json['system_name'],
                                                      numerics=implementation_benchmark_json['weight_data_types'])

                    else: # Keep as v0.5 example to catch any obvious mistakes.

                        # Extract LoadGen revision from the second line of e.g.
                        # "pid": 28660, "tid": 28660, "ts": 8750ns : version : .5a1 @ 61220457de
                        # FIXME: In practice, the revision may be different for accuracy and performance runs
                        # (happened on rpi4 due to a late LoadGen fix). We would prefer to use one from
                        # the performance one, as it may be more critical for performance evaluation.
                        # However, as we only write the checklist from the accuracy run, we are somewhat stuck.
                        loadgen_revision = detail[1].split('@')[1].strip()

                        # FIXME: The actual performance_sample_count can be extracted from the performance run.
                        # Again, this is not available to us here.
                        # We could check in user.conf, but we would need to parse it.
                        performance_sample_count = 1024 if task == 'image-classification' else 256

                        checklist = get_checklist_0_5(division=division, task=task, system=system,
                                                      system_name=system_json['system_name'], category=system_json['status'],
                                                      revision=loadgen_revision, benchmark=benchmark, accuracy_pc=accuracy_pc,
                                                      performance_sample_count=performance_sample_count,
                                                      numerics=implementation_benchmark_json['weight_data_types'])
                    with open(checklist_path, 'w') as checklist_file:
                        checklist_file.writelines(checklist)

#                 # Trace file (should omit trace from v0.5).
#                 trace_json_name = 'mlperf_log_trace.json'
#                 trace_json_path = os.path.join(last_dir, trace_json_name)
#                 with open(trace_json_path, 'w') as trace_json_file:
#                     json.dump(mlperf_log.get('trace',{}), trace_json_file, indent=2)

                # Infer Offline from SingleStream results if requested.
                if infer_offline_from_singlestream and scenario == 'singlestream':
                    results_singlestream_dir = scenario_dir
                    results_offline_dir = results_singlestream_dir.replace('singlestream', 'offline')
                    if not os.path.exists(results_offline_dir):
                        copytree(results_singlestream_dir, results_offline_dir)
                    else:
                        print("Warning: '{}' already exists!".format(results_offline_dir))
                    measurements_singlestream_dir = mscenario_dir
                    measurements_offline_dir = measurements_singlestream_dir.replace('singlestream', 'offline')
                    if not os.path.exists(measurements_offline_dir):
                        copytree(measurements_singlestream_dir, measurements_offline_dir)
                    else:
                        print("Warning: '{}' already exists!".format(measurements_offline_dir))

    return


submitter       = os.environ.get('CK_MLPERF_SUBMISSION_SUBMITTER', 'dividiti')
submitter_desc  = os.environ.get('CK_MLPERF_SUBMISSION_SUBMITTER_DESC', submitter)   # description 'dividiti, Politecnico di Milano' used for a combined submission
repo            = os.environ.get('CK_MLPERF_SUBMISSION_REPO','')
extra_tags      = os.environ.get('CK_MLPERF_SUBMISSION_EXTRA_TAGS','')
repos = [ repo ] if repo != '' else []
for repo_uoa in repos:
    # First, process performance and accuracy data.
    check_experimental_results(repo_uoa, extra_tags=extra_tags,
                               submitter=submitter, submitter_desc=submitter_desc,
                               compliance=False, infer_offline_from_singlestream=False)
    # Then, process compliance data.
    check_experimental_results(repo_uoa, extra_tags=extra_tags,
                               submitter=submitter, submitter_desc=submitter_desc,
                               compliance=True, infer_offline_from_singlestream=False)

print("*" * 100)
truncate_accuracy_log_py = os.path.join(upstream_path, 'tools', 'submission', 'truncate_accuracy_log.py')
# Since we generate submissions from CK entries, no need to keep the original log accuracy files.
backup_dir = '/tmp'
open_org_backup_dir = os.path.join(backup_dir, 'open', submitter)
closed_org_backup_dir = os.path.join(backup_dir, 'closed', submitter)
subprocess.run(['rm', '-rf', open_org_backup_dir])
subprocess.run(['rm', '-rf', closed_org_backup_dir])
# Run the truncate accuracy log script.
truncate_accuracy_log = subprocess.getoutput(
        'python3 {} --submitter {} --input {} --backup {}'
        .format(truncate_accuracy_log_py, submitter, root_dir, backup_dir))
print(truncate_accuracy_log)
# truncate_accuracy_log_name = 'truncate_accuracy_log.txt'
# TODO: Write the script log?

print("*" * 100)
submission_checker_py = os.path.join(upstream_path, 'tools', 'submission', 'submission-checker.py')

open_org_results_dir = os.path.join(root_dir, 'open', submitter, 'results')
closed_org_results_dir = os.path.join(root_dir, 'closed', submitter, 'results')

# # For v0.5, the checker had a weird bug, which may no longer be there.
# # When submitting to open, 'closed/<organization>/results' must have existed on disk.
# # Vice versa, when submitting to closed, 'open/<organization>/results' must have existed on disk.
# # Therefore, we created both directories if they did not exist before invoking the checker.
# subprocess.run(['mkdir', '-p', open_org_results_dir])
# subprocess.run(['mkdir', '-p', closed_org_results_dir])

# Run the submission checker script.
checker_log = subprocess.getoutput(
        'python3 {} --submitter {} --input {}'
        .format(submission_checker_py, submitter, root_dir))
print(checker_log)
checker_log_name = 'submission_checker_log.txt'

# Write the script log under closed/<organization> and/or under open/<organization>.
results_dirs = []
if any_closed_results: results_dirs.append(closed_org_results_dir)
if any_open_results:   results_dirs.append(open_org_results_dir)
for results_dir in results_dirs:
    checker_log_path = os.path.join(results_dir, checker_log_name)
    with open(checker_log_path, 'w') as checker_log_file:
        checker_log_file.write(checker_log)
        print(results_dir)
        print('  |_%s' % checker_log_name)
