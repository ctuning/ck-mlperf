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
from shutil import copy2
from copy import deepcopy


# ### Scientific

# If some of the scientific packages are missing, please install them using:
# ```
# # python3 -m pip install jupyter pandas numpy matplotlib seaborn --user
# ```

# In[ ]:


import IPython as ip
import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb


# In[ ]:


print ('IPython version: %s' % ip.__version__)
print ('Pandas version: %s' % pd.__version__)
print ('NumPy version: %s' % np.__version__)
print ('Matplotlib version: %s' % mp.__version__)
print ('Seaborn version: %s' % sb.__version__)


# In[ ]:


from IPython.display import Image, display
def display_in_full(df):
    pd.options.display.max_columns = len(df.columns)
    pd.options.display.max_rows = len(df.index)
    display(df)


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


default_colormap = cm.autumn
default_fontsize = 16
default_barwidth = 0.8
default_figwidth = 24
default_figheight = 3
default_figdpi = 200
default_figsize = [default_figwidth, default_figheight]


# In[ ]:


if mp.__version__[0]=='2': mp.style.use('classic')
mp.rcParams['figure.max_open_warning'] = 200
mp.rcParams['figure.dpi'] = default_figdpi
mp.rcParams['font.size'] = default_fontsize
mp.rcParams['legend.fontsize'] = 'medium'


# In[ ]:


# FIXME: Do not hardcode - locate via CK.
pythonpath_coco = '/home/anton/CK_TOOLS/tool-coco-master-gcc-8.3.0-compiler.python-3.6.9-linux-64/'
sys.path.append(pythonpath_coco)
from pycocotools.coco import COCO


# ### Collective Knowledge

# If CK is not installed, please install it using:
# ```
# # python -m pip install ck
# ```

# In[ ]:


import ck.kernel as ck
print ('CK version: %s' % ck.__version__)


# <a id="templates"></a>
# ## System templates

# <a id="templates_firefly"></a>
# ### [Firefly-RK3399](http://en.t-firefly.com/product/rk3399/)

# In[ ]:


firefly = {
    "division": "",
    "submitter": "dividiti",
    "status": "available",
    "system_name": "Firefly-RK3399 (firefly)",

    "number_of_nodes": "1",
    "host_processor_model_name": "Arm Cortex-A72 MP2 (big); Arm Cortex-A53 MP4 (LITTLE)",
    "host_processors_per_node": "1",
    "host_processor_core_count": "2 (big); 4 (LITTLE)",
    "host_processor_frequency": "1800 MHz (big), 1400 MHz (LITTLE)",
    "host_processor_caches": "L1I$ 48 KiB, L1D$ 32 KiB, L2$ 1 MiB (big); L1I$ 32 KiB, L1D$ 32 KiB, L2$ 512 KiB (LITTLE)",
    "host_memory_configuration": "-",
    "host_memory_capacity": "4 GiB",
    "host_storage_capacity": "128 GiB",
    "host_storage_type": "SanDisk Extreme microSD",
    "host_processor_interconnect": "-",
    "host_networking": "-",
    "host_networking_topology": "-",

    "accelerators_per_node": "1",
    "accelerator_model_name": "Arm Mali-T860 MP4",
    "accelerator_frequency": "800 MHz",
    "accelerator_host_interconnect": "-",
    "accelerator_interconnect": "-",
    "accelerator_interconnect_topology": "-",
    "accelerator_memory_capacity": "4 GiB (shared with host)",
    "accelerator_memory_configuration": "-",
    "accelerator_on-chip_memories": "-",
    "cooling": "on-board fan",
    "hw_notes": "http://en.t-firefly.com/product/rk3399/; http://opensource.rock-chips.com/wiki_RK3399",

    "framework": "",
    "operating_system": "Ubuntu 16.04.6 LTS; kernel 4.4.77 #554 (Thu Nov 30 11:30:11 HKT 2017)",
    "other_software_stack": "GCC 7.4.0; Python 3.5.2; OpenCL driver 1.2 v1.r13p0-00rel0-git(a4271c9).31ba04af2d3c01618138bef3aed66c2c",
    "sw_notes": "Powered by Collective Knowledge v1.11.1"
}


# <a id="templates_hikey960"></a>
# ### [Linaro HiKey960](https://www.96boards.org/product/hikey960/)

# In[ ]:


hikey960 = {
    "division": "",
    "submitter": "dividiti",
    "status": "available",
    "system_name": "Linaro HiKey960 (hikey960)",

    "number_of_nodes": "1",
    "host_processor_model_name": "Arm Cortex-A73 MP4 (big); Arm Cortex-A53 MP4 (LITTLE)",
    "host_processors_per_node": "1",
    "host_processor_core_count": "4 (big); 4 (LITTLE)",
    "host_processor_frequency": "2362 MHz (big), 1844 MHz (LITTLE)",
    "host_processor_caches": "L1I$ 256=4x64 KiB, L1D$ 256=4x64 KiB, L2$ 2 MiB (big); L1I$ 128=4x32 KiB, L1D$ 128=4x32 KiB, L2$ 1 MiB (LITTLE)",
    "host_memory_configuration": "-",
    "host_memory_capacity": "3 GiB",
    "host_storage_capacity": "128 GiB",
    "host_storage_type": "SanDisk Extreme microSD",
    "host_processor_interconnect": "-",
    "host_networking": "-",
    "host_networking_topology": "-",

    "accelerators_per_node": "1",
    "accelerator_model_name": "Arm Mali-G71 MP8",
    "accelerator_frequency": "800 MHz",
    "accelerator_host_interconnect": "-",
    "accelerator_interconnect": "-",
    "accelerator_interconnect_topology": "-",
    "accelerator_memory_capacity": "3 GiB (shared with host)",
    "accelerator_memory_configuration": "-",
    "accelerator_on-chip_memories": "-",
    "cooling": "small external fan",
    "hw_notes": "http://www.hisilicon.com/en/Products/ProductList/Kirin",

    "framework": "",
    "operating_system": "Debian 9; kernel 4.19.5-hikey #26 (Thu Aug 22 07:58:35 UTC 2019)",
    "other_software_stack": "GCC 7.4.0; Python 3.5.3; OpenCL driver 2.0 v1.r16p0",
    "sw_notes": "Powered by Collective Knowledge v1.11.1"
}


# <a id="templates_mate10pro"></a>
# ### Huawei Mate 10 Pro

# In[ ]:


mate10pro = {
    "division": "",
    "submitter": "dividiti",
    "status": "available",
    "system_name": "Huawei Mate 10 Pro (mate10pro)",

    "number_of_nodes": "1",
    "host_processor_model_name": "Arm Cortex-A73 MP4 (big); Arm Cortex-A53 MP4 (LITTLE)",
    "host_processors_per_node": "1",
    "host_processor_core_count": "4 (big); 4 (LITTLE)",
    "host_processor_frequency": "2360 MHz (big), 1800 MHz (LITTLE)",
    "host_processor_caches": "L1I$ 256=4x64 KiB, L1D$ 256=4x64 KiB, L2$ 2 MiB (big); L1I$ 128=4x32 KiB, L1D$ 128=4x32 KiB, L2$ 1 MiB (LITTLE)",
    "host_memory_configuration": "-",
    "host_memory_capacity": "6 GiB",
    "host_storage_capacity": "128 GiB",
    "host_storage_type": "Flash",
    "host_processor_interconnect": "-",
    "host_networking": "-",
    "host_networking_topology": "-",

    "accelerators_per_node": "1",
    "accelerator_model_name": "Arm Mali-G72 MP12",
    "accelerator_frequency": "850 MHz",
    "accelerator_host_interconnect": "-",
    "accelerator_interconnect": "-",
    "accelerator_interconnect_topology": "-",
    "accelerator_memory_capacity": "6 GiB (shared with host)",
    "accelerator_memory_configuration": "-",
    "accelerator_on-chip_memories": "-",
    "cooling": "phone case",
    "hw_notes": "https://en.wikichip.org/wiki/hisilicon/kirin/970",

    "framework": "",
    "operating_system": "Android 9.1.0.300(C782E5R1P11); kernel 4.9.148 (Sat Jun 29 20:41:06 CST 2019)",
    "other_software_stack": "Android NDK 17c (LLVM 6.0.2); OpenCL driver 2.0 v1.r14p0-00cet0.0416641283c5d6e2d53c163d0ca99357",
    "sw_notes": "Powered by Collective Knowledge v1.11.1"
}


# <a id="templates_rpi4"></a>
# ### Raspberry Pi 4

# In[ ]:


rpi4 = {
    "division": "",
    "submitter": "dividiti",
    "status": "available",
    "system_name": "Raspberry Pi 4 (rpi4)",

    "number_of_nodes": "1",
    "host_processor_model_name": "Arm Cortex-A72 MP4",
    "host_processors_per_node": "1",
    "host_processor_core_count": "4",
    "host_processor_frequency": "1500 MHz",
    "host_processor_caches": "L1I$ 128=4x32 KiB, L1D$ 128=4x32 KiB, L2$ 1 MiB",
    "host_memory_configuration": "-",
    "host_memory_capacity": "4 GiB",
    "host_storage_capacity": "128 GiB",
    "host_storage_type": "SanDisk Extreme Pro microSD",
    "host_processor_interconnect": "-",
    "host_networking": "-",
    "host_networking_topology": "-",

    "accelerators_per_node": "0",
    "accelerator_model_name": "-",
    "accelerator_frequency": "-",
    "accelerator_host_interconnect": "-",
    "accelerator_interconnect": "-",
    "accelerator_interconnect_topology": "-",
    "accelerator_memory_capacity": "-",
    "accelerator_memory_configuration": "-",
    "accelerator_on-chip_memories": "-",
    "cooling": "http://www.raspberrypiwiki.com/index.php/Armor_Case_B",
    "hw_notes": "https://www.raspberrypi.org/products/raspberry-pi-4-model-b/specifications/",

    "framework": "",
    "operating_system": "Raspbian Buster (Debian 10); kernel 4.19.66-v7l+ #1253 (Thu Aug 15 12:02:08 BST 2019)",
    "other_software_stack": "GCC 8.3.0; Python 3.7.3",
    "sw_notes": "Powered by Collective Knowledge v1.11.1"
}


# <a id="templates_velociti"></a>
# ### HP Z640 workstation

# In[ ]:


velociti = {
    "division": "",
    "submitter": "dividiti",
    "status": "available",
    "system_name": "HP Z640 G1X62EA workstation (velociti)",

    "number_of_nodes": "1",
    "host_processor_model_name": "Intel Xeon CPU E5-2650 v3",
    "host_processors_per_node": "1",
    "host_processor_core_count": "10",
    "host_processor_frequency": "2300 MHz (base); 3000 MHz (turbo)",
    "host_processor_caches": "L1I$ 10x32 KiB, L1D$ 10x32 KiB; L2$ 10x256 KiB; L3$ 25 MiB",
    "host_memory_configuration": "DDR4 (max bandwidth 68 GB/s)",
    "host_memory_capacity": "32 GiB",
    "host_storage_capacity": "512 GiB",
    "host_storage_type": "SSD",
    "host_processor_interconnect": "-",
    "host_networking": "-",
    "host_networking_topology": "-",

    "accelerators_per_node": "1",
    "accelerator_model_name": "NVIDIA GeForce GTX 1080",
    "accelerator_frequency": "1607 MHz (base); 1733 MHz (boost)",
    "accelerator_host_interconnect": "-",
    "accelerator_interconnect": "-",
    "accelerator_interconnect_topology": "-",
    "accelerator_memory_capacity": "8 GiB",
    "accelerator_memory_configuration": "GDDR5X (max bandwidth 320 GB/s)",
    "accelerator_on-chip_memories": "20x48 KiB",
    "cooling": "standard",
    "hw_notes": "The Intel CPU has reached its end-of-life (EOL). http://h20195.www2.hp.com/v2/default.aspx?cc=ie&lc=en&oid=7528701; https://ark.intel.com/products/81705/Intel-Xeon-Processor-E5-2650-v3-25M-Cache-2_30-GHz; http://www.cpu-world.com/CPUs/Xeon/Intel-Xeon%20E5-2650%20v3.html; http://www.geforce.co.uk/hardware/10series/geforce-gtx-1080/",
    
    "framework": "TensorFlow v1.14",
    "operating_system": "Ubuntu 16.04.6 LTS; kernel 4.4.0-112-generic #135-Ubuntu SMP (Fri Jan 19 11:48:36 UTC 2018)",
    "other_software_stack": "Driver 430.50; CUDA 10.1; TensorRT 5.1.5; Docker 19.03.3 (build a872fc2); GCC 7.4.0; Python 3.5.2",
    "sw_notes": "Powered by Collective Knowledge v1.11.4"
}


# <a id="templates_default"></a>
# ### Default

# In[ ]:


# Default `system_desc_id.json` (to catch uninitialized descriptions)
default_system_json = {
    "division": "reqired",
    "submitter": "required",
    "status": "required",
    "system_name": "required",

    "number_of_nodes": "required",
    "host_processor_model_name": "required",
    "host_processors_per_node": "required",
    "host_processor_core_count": "required",
    "host_processor_frequency": "",
    "host_processor_caches": "",
    "host_memory_configuration": "",
    "host_memory_capacity": "required",
    "host_storage_capacity": "required",
    "host_storage_type": "required",
    "host_processor_interconnect": "",
    "host_networking": "",
    "host_networking_topology": "",

    "accelerators_per_node": "required",
    "accelerator_model_name": "required",
    "accelerator_frequency": "",
    "accelerator_host_interconnect": "",
    "accelerator_interconnect": "",
    "accelerator_interconnect_topology": "",
    "accelerator_memory_capacity": "required",
    "accelerator_memory_configuration": "",
    "accelerator_on-chip_memories": "",
    "cooling": "",
    "hw_notes": "",

    "framework": "required",
    "operating_system": "required",
    "other_software_stack": "required",
    "sw_notes": ""
}


# <a id="systems"></a>
# ## Systems

# In[ ]:


# Generate division_systems dictionary.
division_systems = {}

platform_templates = {
    'firefly'   : firefly,
    'hikey960'  : hikey960,
    'mate10pro' : mate10pro,
    'rpi4'      : rpi4,
    'velociti'  : velociti
}

divisions = [ 'open', 'closed' ]
platforms = [ 'firefly', 'hikey960', 'mate10pro', 'rpi4', 'velociti' ]
for division in divisions:
    for platform in platforms:
        if platform == 'velociti':
            libraries = [ 'tensorflow-v1.14' ]
        elif platform == 'mate10pro':
            libraries = [ 'tflite-v1.13', 'armnn-v19.08' ]
        else:
            libraries = [ 'tflite-v1.15', 'armnn-v19.08' ]
        for library in libraries:
            if library == 'armnn-v19.08':
                if platform == 'rpi4':
                    backends = [ 'neon' ]
                else:
                    backends = [ 'neon', 'opencl' ]
                library_backends = [ library+'-'+backend for backend in backends ]
            elif library == 'tensorflow-v1.14':
                backends = [ 'cpu', 'cuda', 'tensorrt', 'tensorrt-dynamic' ]
                library_backends = [ library+'-'+backend for backend in backends ]
            else:
                library_backends = [ library ]
            for library_backend in library_backends:
                division_system = division+'-'+platform+'-'+library_backend
                frameworks = {
                    'armnn-v19.08-opencl' : 'ArmNN v19.08 (OpenCL)',
                    'armnn-v19.08-neon' : 'ArmNN v19.08 (Neon)',
                    'tflite-v1.13': 'TFLite v1.13.1',
                    'tflite-v1.15': 'TFLite v1.15.0-rc2',
                    'tensorflow-v1.14-cpu': 'TensorFlow v1.14 (CPU)',
                    'tensorflow-v1.14-cuda': 'TensorFlow v1.14 (CUDA)',
                    'tensorflow-v1.14-tensorrt': 'TensorFlow v1.14 (TensorRT-static)',
                    'tensorflow-v1.14-tensorrt-dynamic': 'TensorFlow v1.14 (TensorRT-dynamic)',
                }
                template = deepcopy(platform_templates[platform])
                template.update({
                    'division'  : division,
                    'submitter' : 'dividiti', # 'dividiti' if platform != 'velociti' else 'dividiti, Politecnico di Milano'
                    'status'    : 'available' if library_backend != 'tensorflow-v1.14-cpu' else 'RDI',
                    'framework' : frameworks[library_backend]
                })
                if (not library_backend.startswith('tensorflow') and not library_backend.endswith('opencl'))                 or library_backend.endswith('cpu'):
                    template.update({
                        'accelerator_frequency' : '-',
                        'accelerator_memory_capacity' : '-',
                        'accelerator_memory_configuration': '-',
                        'accelerator_model_name' : '-',
                        'accelerator_on-chip_memories': '-',
                        'accelerators_per_node' : '0',
                    })
                division_systems[division_system] = template
                print("=" * 100)
                print(division_system)
                print("=" * 100)
                pprint(template)
                print("-" * 100)
                print("")


# <a id="implementations"></a>
# ## Implementations

# ### Image classification

# In[ ]:


# Generate implementation_benchmarks dictionary.
implementation_benchmarks = {}

# Default `system_desc_id_imp.json` (to catch uninitialized descriptions)
default_implementation_benchmark_json = {
    "input_data_types": "required",
    "retraining": "required",
    "starting_weights_filename": "required",
    "weight_data_types": "required",
    "weight_transformations": "required"
}

# For each image classification implementation.
for implementation in [ 'image-classification-tflite', 'image-classification-armnn-tflite' ]:
    # Add MobileNet.
    implementation_mobilenet = implementation+'-'+'mobilenet'
    implementation_benchmarks[implementation_mobilenet] = {
        "input_data_types": "fp32",
        "weight_data_types": "fp32",
        "retraining": "no",
        "starting_weights_filename": "https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz",
        "weight_transformations": "TFLite"
    }
    # Add MobileNet quantized.
    implementation_mobilenet_quantized = implementation+'-'+'mobilenet-quantized'
    implementation_benchmarks[implementation_mobilenet_quantized] = {
        "input_data_types": "uint8",
        "weight_data_types": "uint8",
        "retraining": "no",
        "starting_weights_filename": "https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz",
        "weight_transformations": "TFLite"
    }
    # Add ResNet.
    implementation_resnet = implementation+'-'+'resnet'
    implementation_benchmarks[implementation_resnet] = {
        "input_data_types": "fp32",
        "weight_data_types": "fp32",
        "retraining": "no",
        "starting_weights_filename": "https://zenodo.org/record/2535873/files/resnet50_v1.pb",
        "weight_transformations": "TF -> TFLite"
    }
    # Add any MobileNets-v1,v2 model.
    def add_implementation_mobilenet(implementation_benchmarks, version, multiplier, resolution, quantized=False):
        base_url = 'https://zenodo.org/record/2269307/files' if version == 1 else 'https://zenodo.org/record/2266646/files'
        url = '{}/mobilenet_v{}_{}_{}{}.tgz'.format(base_url, version, multiplier, resolution, '_quant' if quantized else '')
        benchmark = 'mobilenet-v{}-{}-{}{}'.format(version, multiplier, resolution, '-quantized' if quantized else '')
        if quantized and (version != 1 or implementation != 'image-classification-tflite'):
            return
        if implementation == 'image-classification-tflite':
            weights_transformations = 'TFLite'
        elif implementation == 'image-classification-armnn-tflite':
            weights_transformations = 'TFLite -> ArmNN'
        else:
            raise "Unknown implementation '%s'!" % implementation
        implementation_benchmark = implementation+'-'+benchmark
        implementation_benchmarks[implementation_benchmark] = {
            "input_data_types": "uint8" if quantized else "fp32",
            "weight_data_types": "uint8" if quantized else "fp32",
            "retraining": "no",
            "starting_weights_filename": url,
            "weight_transformations": weights_transformations
        }
        return
    # MobileNet-v1.
    version = 1
    for multiplier in [ 1.0, 0.75, 0.5, 0.25 ]:
        for resolution in [ 224, 192, 160, 128 ]:
            add_implementation_mobilenet(implementation_benchmarks, version, multiplier, resolution, quantized=False)
            add_implementation_mobilenet(implementation_benchmarks, version, multiplier, resolution, quantized=True)
    # MobileNet-v2.
    version = 2
    for multiplier in [ 1.0, 0.75, 0.5, 0.35 ]:
        for resolution in [ 224, 192, 160, 128, 96 ]:
            add_implementation_mobilenet(implementation_benchmarks, version, multiplier, resolution)
    add_implementation_mobilenet(implementation_benchmarks, version=2, multiplier=1.3, resolution=224)
    add_implementation_mobilenet(implementation_benchmarks, version=2, multiplier=1.4, resolution=224)


# ### Object detection

# In[ ]:


object_detection_benchmarks = {
    'rcnn-nas-lowproposals' : {
        "name" : "Faster-RCNN-NAS lowproposals",
        "url" : "http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz",
        "width" : 1200,
        "height" : 1200,
    },
    'rcnn-resnet50-lowproposals' : {
        "name" : "Faster-RCNN-ResNet50 lowproposals",
        "url" : "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz",
        "width" : 1024,
        "height" : 600,
    },
    'rcnn-resnet101-lowproposals' : {
        "name" : "Faster-RCNN-ResNet101 lowproposals",
        "url" : "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz",
        "width" : 1024,
        "height" : 600,
    },
    'rcnn-inception-resnet-v2-lowproposals' : {
        "name" : "Faster-RCNN-Inception-ResNet-v2 lowproposals",
        "url" : "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz",
        "width" : 1024,
        "height" : 600,
    },
    'rcnn-inception-v2' : {
        "name" : "Faster-RCNN Inception-v2",
        "url" : "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz",
        "width" : 1024,
        "height" : 600,
    },
    'ssd-inception-v2' : {
        "name" : "SSD-Inception-v2",
        "url" : "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz",
        "width" : 300,
        "height" : 300,
    },
    'ssd-mobilenet-v1-quantized-mlperf' : {
        "name" : "MLPerf SSD-MobileNet",
        "url" : "https://zenodo.org/record/3361502/files/ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
        "width" : 300,
        "height" : 300,
        "provenance" : "Google",
    },
    'ssd-mobilenet-v1-non-quantized-mlperf' : {
        "name" : "MLPerf SSD-MobileNet quantized",
        "url" : "https://zenodo.org/record/3252084/files/mobilenet_v1_ssd_8bit_finetuned.tar.gz",
        "width" : 300,
        "height" : 300,
        "provenance" : "Habana"
    },
    'ssd-mobilenet-v1-fpn' : {
        "name" : "SSD-MobileNet-v1 FPN SBP",
        "url" : "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz",
        "width" : 640,
        "height" : 640,
    },
    'ssd-resnet50-fpn' : {
        "name" : "SSD-ResNet50-v1 FPN SBP",
        "url" : "http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz",
        "width" : 640,
        "height" : 640,
    },
    'ssdlite-mobilenet-v2' : {
        "name" : "SSDLite-MobileNet-v2",
        "url" : "http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz",
        "width" : 300,
        "height" : 300,
    },
    'yolo-v3' : {
        "name" : "YOLO-v3",
        "url" : "https://zenodo.org/record/3386327/files/yolo_v3_coco.tar.gz",
        "width" : 416,
        "height" : 416,
        "provenance" : "https://github.com/YunYang1994/tensorflow-yolov3/"
    }
}
    
# For each object detection implementation.
for implementation in [ 'mlperf-inference-vision' ]:
    for benchmark in object_detection_benchmarks.keys():
        implementation_benchmark = implementation+'-'+benchmark
        implementation_benchmarks[implementation_benchmark] = {
            "input_data_types": "fp32",
            "weight_data_types": "fp32",
            "retraining": "no",
            "starting_weights_filename": object_detection_benchmarks[benchmark]['url'],
#            "name" : object_detection_benchmarks[benchmark]['name'], # submission checker complains about "unknwon field name"
            "weight_transformations": "None (TensorFlow)"
        }

# from pprint import pprint
# pprint(implementation_benchmarks)


# In[ ]:


implementation_readmes = {}
implementation_readmes['image-classification-tflite'] = """# MLPerf Inference - Image Classification - TFLite

This C++ implementation uses TFLite to run TFLite models for Image Classification on CPUs.

## Links
- [Jupyter notebook](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb)
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-tflite-loadgen).
- [Instructions](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite/README.md).
"""

implementation_readmes['image-classification-armnn-tflite'] = """# MLPerf Inference - Image Classification - ArmNN-TFLite

This C++ implementation uses ArmNN with the TFLite frontend to run TFLite models for Image Classification on Arm Cortex CPUs and Arm Mali GPUs.

## Links
- [Jupyter notebook](https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/1xlv5oacgobrfd4/mlperf-inference-v0.5-dividiti.ipynb)
- [Source code](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-armnn-tflite-loadgen).
- [Instructions](https://github.com/ARM-software/armnn-mlperf/blob/master/README.md).
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


# In[ ]:


implementation_paths = {}
for implementation in [ 'image-classification-tflite', 'image-classification-armnn-tflite', 'mlperf-inference-vision' ]:
    implementation_uoa = implementation
    if implementation.startswith('image-classification'):
        implementation_uoa += '-loadgen'
        repo_uoa = 'ck-mlperf'
    else: # TODO: move to ck-mlperf, then no need for special case.
        repo_uoa = 'ck-object-detection'
    r = ck.access({'action':'find', 'repo_uoa':repo_uoa, 'module_uoa':'program', 'data_uoa':implementation_uoa})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)
    implementation_paths[implementation] = r['path']


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


checklist_template = """MLPerf Inference 0.5 Self-Certification Checklist

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

def get_checklist(checklist_template=checklist_template, name='Anton Lokhmotov', email='anton@dividiti.com',
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
        # Division.
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

# null = get_checklist(system='rpi4-armnn-v19.08-neon', system_name='Raspberry Pi 4 (rpi4)', benchmark='mobilenet', accuracy_pc=70.241, numerics='uint8')
# null = get_checklist(system='hikey960-tflite-v1.15', system_name='Linaro HiKey 960 (hikey960)', benchmark='resnet', accuracy_pc=75.692, revision='deadbeef')
null = get_checklist(system='velociti-tensorflow-v1.14-cpu', name='Anton Lokhmotov; Emanuele Vitali', email='anton@dividiti.com; emanuele.vitali@polimi.it', system_name='HP Z640 G1X62EA workstation (velociti)', division='open', category='RDI', benchmark='ssd-mobilenet-fpn')


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


# Locate upstream master.
# r = ck.access({'action':'locate', 'module_uoa':'env', 'tags':'mlperf,inference,source,upstream.master'})
# Locate variation with audit test fixes.
r = ck.access({'action':'locate', 'module_uoa':'env', 'tags':'mlperf,inference,source,upstream.pr518'})
if r['return']>0:
    print('Error: %s' % r['error'])
    exit(1)
# Pick any source location and look under 'inference/v0.5/mlperf.conf'.
upstream_path = os.path.join(list(r['install_locations'].values())[0], 'inference')
upstream_path


# In[ ]:


def check_experimental_results(repo_uoa, module_uoa='experiment', tags='mlperf', submitter='dividiti', path=None, audit=False):
    if not path:
        path_list = get_ipython().getoutput('ck find repo:$repo_uoa')
        path = path_list[0]
    root_dir = os.path.join(path, 'submissions_inference_0_5')
    if not os.path.exists(root_dir): os.mkdir(root_dir)
    print("Storing results under '%s'" % root_dir)
    
    r = ck.access({'action':'search', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'tags':tags})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)
    experiments = r['lst']

    for experiment in experiments:
        data_uoa = experiment['data_uoa']
        r = ck.access({'action':'list_points', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
        if r['return']>0:
            print('Error: %s' % r['error'])
            exit(1)
        print("*" * 100)

        tags = r['dict']['tags']
        #print(tags)
        backend = ''
        preprocessing = ''
        if 'velociti' in tags:
            # Expected format: [ "mlperf", "open", "object-detection", "velociti", "cpu", "rcnn-inception-resnet-v2-lowproposals", "singlestream", "accuracy" ]
            (_, division, task, platform, backend, benchmark, scenario, mode) = tags
            library = 'tensorflow-v1.14'
        elif 'accuracy' in tags:
            # FIXME: With the benefit of hindsight, [ ..., "armnn-v19.08", "neon", ... ] should have come 
            # as one tag "armnn-v19.08-neon", since we join them in this notebook anyway.
            if 'neon' in tags or 'opencl' in tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "accuracy", "using-opencv" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode, preprocessing) = tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "accuracy", "using-opencv" ]
                (_, division, task, platform, library, benchmark, scenario, mode, preprocessing) = tags
        elif 'performance' in tags:            
            if 'neon' in tags or 'opencl' in tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "performance" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode) = tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "performance" ]
                (_, division, task, platform, library, benchmark, scenario, mode) = tags
        elif 'audit' in tags: # As accuracy but with the test name instead of the preprocessing method.
            if 'neon' in tags or 'opencl' in tags:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "armnn-v19.08", "neon", "mobilenet-v1-0.5-128", "singlestream", "audit", "TEST03" ]
                (_, division, task, platform, library, backend, benchmark, scenario, mode, test) = tags
            else:
                # Expected format: [ "mlperf", "open", "image-classification", "firefly", "tflite-v1.15", "mobilenet-v1-0.5-128", "singlestream", "audit", "TEST03" ]
                (_, division, task, platform, library, benchmark, scenario, mode, test) = tags
        else:
            raise "Expected 'accuracy' or 'performance' or 'audit' in tags!"

#         if mode == 'accuracy': continue
            
        organization = submitter

        if backend != '':
            system = platform+'-'+library+'-'+backend
        else:
            system = platform+'-'+library
        division_system = division+'-'+system

        if library.startswith('tflite'):
            implementation = task+'-tflite'
        elif library.startswith('armnn'):
            implementation = task+'-armnn-tflite'
        else: # Official app with CK adaptations.
            implementation = 'mlperf-inference-vision'
        implementation_benchmark = implementation+'-'+benchmark
        
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
        with open(system_json_path, 'w') as system_json_file:
#             pprint(division_system)
#             pprint(division_systems)
            system_json = division_systems.get(division_system, default_system_json)
            json.dump(system_json, system_json_file, indent=2)
            print('%s' % systems_dir)
            if system_json == default_system_json:
                print('  |_ %s [DEFAULT]' % system_json_name)
                raise
            else:
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
        implementation_dir = os.path.join(benchmark_dir, implementation)
        if not os.path.exists(implementation_dir): os.mkdir(implementation_dir)
        print('%s' % code_dir)

        # Create 'README.md'.
        implementation_readme_name = 'README.md'
        implementation_readme_path = os.path.join(implementation_dir, implementation_readme_name)
#         pprint(implementation)
#         pprint(implementation_readmes)
        implementation_readme = implementation_readmes.get(implementation, '')
        with open(implementation_readme_path, 'w') as implementation_readme_file:
            implementation_readme_file.writelines(implementation_readme)
        if implementation_readme == '':
            print('  |_ %s [EMPTY]' % implementation_readme_name)
            raise
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
        scenario_dir = os.path.join(benchmark_dir, scenario)
        if not os.path.exists(scenario_dir): os.mkdir(scenario_dir)
        print(scenario_dir)
        
        # Create '<system_desc_id>_<implementation_id>.json'.
        system_implementation_json_name = system+'_'+implementation+'.json'
        system_implementation_json_path = os.path.join(scenario_dir, system_implementation_json_name)
        with open(system_implementation_json_path, 'w') as system_implementation_json_file:
            implementation_benchmark_json = implementation_benchmarks.get(implementation_benchmark, default_implementation_benchmark_json)
            if implementation_benchmark_json != default_implementation_benchmark_json:
                print('  |_ %s [for %s]' % (system_implementation_json_name, implementation_benchmark))
                json.dump(implementation_benchmark_json, system_implementation_json_file, indent=2)
            else:
                print('  |_ %s [DEFAULT]' % system_implementation_json_name)
                raise "Default implementation!"
        
        # Create 'README.md' based on the division and task (basically, mentions a division- and task-specific script).
        measurements_readme_name = 'README.md'
        measurements_readme_path = os.path.join(scenario_dir, measurements_readme_name)
        measurements_readme = measurements_readmes.get(division+'-'+task, '')
        if measurements_readme != '':
            with open(measurements_readme_path, 'w') as measurements_readme_file:
                measurements_readme_file.writelines(measurements_readme)
            print('  |_ %s [for %s %s]' % (measurements_readme_name, division, task))
        else:
            raise "Invalid measurements README!"
        
        # Copy 'user.conf' from implementation source.
        user_conf_name = 'user.conf'
        implementation_path = implementation_paths.get(implementation, '')
#         pprint(implementation)
#         pprint(implementation_paths)
        if implementation_path != '':
            user_conf_path = os.path.join(implementation_path, user_conf_name)
            copy2(user_conf_path, scenario_dir)
            print('  |_ %s [from %s]' % (user_conf_name, user_conf_path))
        else:
            raise "Invalid implementation path!"
        
        # Copy 'mlperf.conf' from MLPerf Inference source.
        mlperf_conf_name = 'mlperf.conf'
        mlperf_conf_path = os.path.join(scenario_dir, mlperf_conf_name)
        if implementation in [ 'image-classification-tflite', 'image-classification-armnn-tflite' ]:
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
        #         compliance_checker_log.txt
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
        print(mode_dir)
        
        if audit:
            # Deal with a subset of audit tests.
#             if test not in [ 'TEST03' ]: # [ 'TEST01', 'TEST03', 'TEST04-A', 'TEST04-B', 'TEST05' ]:
#                 continue
            # Save the accuracy and performance dirs for the corresponding submission experiment.
            accuracy_dir = os.path.join(scenario_dir, 'accuracy')
            performance_dir = os.path.join(scenario_dir, 'performance', 'run_1')
            # Use the mode expected for each test.
            mode = 'performance' if test != 'TEST03' else 'submission'
            # Create a similar directory structure to results_dir, with another level, test_dir,
            # between scenario_dir and mode_dir.
            audit_dir = os.path.join(organization_dir, 'audit')
            if not os.path.exists(audit_dir): os.mkdir(audit_dir)
            system_dir = os.path.join(audit_dir, system)
            if not os.path.exists(system_dir): os.mkdir(system_dir)
            benchmark_dir = os.path.join(system_dir, benchmark)
            if not os.path.exists(benchmark_dir): os.mkdir(benchmark_dir)
            scenario_dir = os.path.join(benchmark_dir, scenario)
            if not os.path.exists(scenario_dir): os.mkdir(scenario_dir)
            test_dir = os.path.join(scenario_dir, test)
            if not os.path.exists(test_dir): os.mkdir(test_dir)
            mode_dir = os.path.join(test_dir, mode)
            if not os.path.exists(mode_dir): os.mkdir(mode_dir)

        # For each point (should be one point for each performance run).
        points = r['points']
        for (point, point_idx) in zip(points, range(1,len(points)+1)):
            point_file_path = os.path.join(r['path'], 'ckp-%s.0001.json' % point)
            with open(point_file_path) as point_file:
                point_data_raw = json.load(point_file)
            characteristics_list = point_data_raw['characteristics_list']
            characteristics = characteristics_list[0]

            # Set the leaf directory.
            if mode == 'performance':
                run_dir = os.path.join(mode_dir, 'run_%d' % point_idx)
                if not os.path.exists(run_dir): os.mkdir(run_dir)
                last_dir = run_dir
            else:
                last_dir = mode_dir
            print(last_dir)

            # Dump files in the leaf directory.
            mlperf_log = characteristics['run'].get('mlperf_log',{})
            # Summary file (with errors and warnings in accuracy mode, with statistics in performance mode).
            summary_txt_name = 'mlperf_log_summary.txt'
            summary_txt_path = os.path.join(last_dir, summary_txt_name)
            summary = mlperf_log.get('summary','')
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
            # TODO: Move the next 5 lines into the (if mode == 'accuracy') block,
            # once the submission checker no longer complains as follows:
            # "performance/run_1 has file list mismatch (['mlperf_log_accuracy.json'])"
            accuracy_json_name = 'mlperf_log_accuracy.json'
            accuracy_json_path = os.path.join(last_dir, accuracy_json_name)
            with open(accuracy_json_path, 'w') as accuracy_json_file:
                json.dump(mlperf_log.get('accuracy',{}), accuracy_json_file, indent=2)
                print('  |_ %s' % accuracy_json_name)
            # Do what's required by NVIDIA's audit tests.
            if audit:
                test_path = os.path.join(upstream_path, 'v0.5', 'audit', 'nvidia', test)
                if 'TEST01' in tags:
                    # Verify that the accuracy (partially) dumped for the audit test matches that for the submision.
                    verify_accuracy_py = os.path.join(test_path, 'verify_accuracy.py')
                    submission_accuracy_json_path = os.path.join(accuracy_dir, accuracy_json_name)
                    verify_accuracy_txt = get_ipython().getoutput('python3 $verify_accuracy_py -a $submission_accuracy_json_path -p $accuracy_json_path')
                    verify_accuracy_txt_name = 'verify_accuracy.txt'
                    verify_accuracy_txt_path = os.path.join(test_dir, verify_accuracy_txt_name)
                    with open(verify_accuracy_txt_path, 'w') as verify_accuracy_txt_file:
                        verify_accuracy_txt_file.writelines('\n'.join(verify_accuracy_txt))
                        print('%s' % test_dir)
                        print('  |_ %s' % verify_accuracy_txt_name)
                if test in [ 'TEST01', 'TEST03', 'TEST05' ]:
                    # Verify that the performance for the audit test matches that for the submission.
                    verify_performance_py = os.path.join(test_path, 'verify_performance.py')
                    submission_summary_txt_path = os.path.join(performance_dir, summary_txt_name)
                    verify_performance_txt = get_ipython().getoutput('python3 $verify_performance_py -r $submission_summary_txt_path -t $summary_txt_path')
                    verify_performance_txt_name = 'verify_performance.txt'
                    verify_performance_txt_path = os.path.join(test_dir, verify_performance_txt_name)
                    with open(verify_performance_txt_path, 'w') as verify_performance_txt_file:
                        verify_performance_txt_file.writelines('\n'.join(verify_performance_txt))
                        print('%s' % test_dir)
                        print('  |_ %s' % verify_performance_txt_name)
                if test in [ 'TEST04-A', 'TEST04-B' ]:
                    test04a_summary_txt_path = os.path.join(scenario_dir, 'TEST04-A', 'performance', 'run_1', summary_txt_name)
                    test04b_summary_txt_path = os.path.join(scenario_dir, 'TEST04-B', 'performance', 'run_1', summary_txt_name)
                    if os.path.exists(test04a_summary_txt_path) and os.path.exists(test04b_summary_txt_path):
                        # If both tests have been processed, verify that their performance matches.
                        verify_performance_py = os.path.join(upstream_path, 'v0.5', 'audit', 'nvidia', 'TEST04-A', 'verify_test4_performance.py')
                        #print("python3 {} -u {} -s {}".format(verify_performance_py, test04a_summary_txt_path, test04b_summary_txt_path))
                        verify_performance_txt = get_ipython().getoutput('python3 $verify_performance_py -u $test04a_summary_txt_path -s $test04b_summary_txt_path')
                        #print(verify_performance_txt)
                        verify_performance_txt_name = 'verify_performance.txt'
                        verify_performance_txt_path = os.path.join(scenario_dir, 'TEST04-A', verify_performance_txt_name)
                        with open(verify_performance_txt_path, 'w') as verify_performance_txt_file:
                            verify_performance_txt_file.writelines('\n'.join(verify_performance_txt))
                            print('%s' % os.path.join(scenario_dir, 'TEST04-A'))
                            print('  |_ %s' % verify_performance_txt_name)
                    else:
                        # Need both A/B tests to be processed. Wait for the other one.
                        continue
            # Generate accuracy.txt.
            if mode == 'accuracy' or mode == 'submission':
                accuracy_txt_name = 'accuracy.txt'
                accuracy_txt_path = os.path.join(last_dir, accuracy_txt_name)
                if task == 'image-classification':
                    accuracy_imagenet_py = os.path.join(upstream_path, 'v0.5', 'classification_and_detection', 'tools', 'accuracy-imagenet.py')
                    imagenet_val_file = '$HOME/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt' # FIXME: Do not hardcode - locate via CK.
                    accuracy_txt = get_ipython().getoutput('python3 $accuracy_imagenet_py --imagenet-val-file $imagenet_val_file --mlperf-accuracy-file $accuracy_json_path')
                    # The last (and only line) is e.g. "accuracy=76.442%, good=38221, total=50000".
                    accuracy_line = accuracy_txt[-1]
                    match = re.match('accuracy=(.+)%, good=(\d+), total=(\d+)', accuracy_line)
                    accuracy_pc = float(match.group(1))
                elif task == 'object-detection':
                    accuracy_coco_py = os.path.join(upstream_path, 'v0.5', 'classification_and_detection', 'tools', 'accuracy-coco.py')
                    coco_dir = '/home/anton/CK_TOOLS/dataset-coco-2017-val' # FIXME: Do not hardcode - locate via CK.
                    os.environ['PYTHONPATH'] = pythonpath_coco+':'+os.environ.get('PYTHONPATH','')
                    accuracy_txt = get_ipython().getoutput('python3 $accuracy_coco_py --coco-dir $coco_dir --mlperf-accuracy-file $accuracy_json_path')
                    # The last line is e.g. "mAP=13.323%".
                    accuracy_line = accuracy_txt[-1]
                    match = re.match('mAP\=([\d\.]+)\%', accuracy_line)
                    accuracy_pc = float(match.group(1))
                else:
                    raise "Invalid task '%s'!" % task
                with open(accuracy_txt_path, 'w') as accuracy_txt_file:
                    accuracy_txt_file.writelines('\n'.join(accuracy_txt))
                    print('  |_ %s [%.3f%% parsed from "%s"]' % (accuracy_txt_name, accuracy_pc, accuracy_line))
            # Generate submission_checklist.txt for each system, benchmark and scenario under "measurements/".
            if mode == 'accuracy' and not audit:
                checklist_name = 'submission_checklist.txt'
                checklist_path = os.path.join(measurements_dir, system, benchmark, scenario, checklist_name)
                system_json = division_systems.get(division_system, default_system_json)

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
                # Write the checklist.
                if division == 'open' and task == 'object-detection':
                    # Collaboration between dividiti and Politecnico di Milano.
                    print(system)
                    checklist = get_checklist(name='Anton Lokhmotov; Emanuele Vitali',
                                              email='anton@dividiti.com; emanuele.vitali@polimi.it',
                                              division=division, task=task, system=system,
                                              system_name=system_json['system_name'], category=system_json['status'],
                                              revision=loadgen_revision, benchmark=benchmark, accuracy_pc=accuracy_pc,
                                              performance_sample_count=performance_sample_count,
                                              numerics=implementation_benchmark_json['weight_data_types'])
                else:
                    checklist = get_checklist(division=division, task=task, system=system,
                                              system_name=system_json['system_name'], category=system_json['status'],
                                              revision=loadgen_revision, benchmark=benchmark, accuracy_pc=accuracy_pc,
                                              performance_sample_count=performance_sample_count,
                                              numerics=implementation_benchmark_json['weight_data_types'])
                with open(checklist_path, 'w') as checklist_file:
                    checklist_file.writelines(checklist)

#             # Trace file (should omit trace from v0.5).
#             trace_json_name = 'mlperf_log_trace.json'
#             trace_json_path = os.path.join(last_dir, trace_json_name)
#             with open(trace_json_path, 'w') as trace_json_file:
#                 json.dump(mlperf_log.get('trace',{}), trace_json_file, indent=2)
    return


# In[ ]:


# The path is where mlperf/submissions_inference_0_5 is cloned under.
path = '/home/anton/projects/mlperf/'
submitter = 'dividiti'


# ### Extract submission repos

# In[ ]:


# # repos = repos_image_classification_closed + repos_image_classification_open + repos_object_detection_open
# repos = [ 'mlperf.open.image-classification.firefly.tflite-v1.15.mobilenet-v1-quantized' ]
# for repo_uoa in repos:
#     check_experimental_results(repo_uoa, path=path, submitter=submitter, audit=False)


# ### Extract audit repos

# In[ ]:


# # audit_repos = repos_image_classification_closed_audit + repos_image_classification_open_audit
# audit_repos = [ 'mlperf.closed.image-classification.mate10pro.audit' ]
# for repo_uoa in audit_repos:
#     check_experimental_results(repo_uoa, path=path, submitter=submitter, audit=True)


# ### Run submission checker

# In[ ]:


print("*" * 100)
submission_checker_py = os.path.join(upstream_path, 'v0.5', 'tools', 'submission', 'submission-checker.py')
# The checker has a weird bug. When submitting to open, 'closed/<organization>/results' must exist on disk.
# Vice versa, When submitting to closed, 'open/<organization>/results' must exist on disk. 
# Therefore, create both directories if they do not exist before invoking the checker.
root_dir = os.path.join(path, 'submissions_inference_0_5')
open_org_results_dir = os.path.join(root_dir, 'open', submitter, 'results')
closed_org_results_dir = os.path.join(root_dir, 'closed', submitter, 'results')
get_ipython().system('mkdir -p $open_org_results_dir')
get_ipython().system('mkdir -p $closed_org_results_dir')
# Run the checker.
checker_log = get_ipython().getoutput('python3 $submission_checker_py --input $root_dir --submitter $submitter')
checker_log = "\n".join(checker_log)
print(checker_log)
checker_log_name = 'compliance_checker_log.txt'
# Write the checker results once closed/<organization> and once under open/<organization>.
for results_dir in [ open_org_results_dir, closed_org_results_dir ]:
    checker_log_path = os.path.join(results_dir, checker_log_name)
    with open(checker_log_path, 'w') as checker_log_file:
        checker_log_file.writelines(checker_log)
        print(results_dir)
        print('  |_%s' % checker_log_name)

