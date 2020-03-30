#!/usr/bin/env python
# coding: utf-8

# # [MLPerf Inference Results v0.5](https://github.com/mlperf/inference/tree/master/v0.5)
# ## Automatic results table generation (c) [dividiti](http://dividiti.com/)

# ## Includes

# In[ ]:


import os
import re
import json
from sys import exit

# In[ ]:


import IPython as ip
import pandas as pd
import numpy as np


# In[ ]:


# ## Definitions

# ### Path to the repository with results

# In[ ]:


# Clone the results directory:
# git clone https://github.com/mlperf/inference_results_v0.5 <results_path>
# or
# git clone https://github.com/dividiti/inference_results_v0.5 <results_path>
# results_path = '/home/anton/projects/mlperf/inference_results_v0.5_dividiti'
results_path = os.environ.get('CK_MLPERF_SUBMISSION_ROOT', '')
if results_path == '':
    print("Set CK_MLPERF_SUBMISSION_ROOT to the root of a submissions directory!")
    exit(1)


# ### Path to the cache

# In[ ]:

cache_compression = 'zip'
cache_protocol = 2 # Supported since Python 2.3

cache_file = os.environ.get('CK_MLPERF_DASHBOARD_FILE', 'mlperf-inference-results.zip')
cache_dir = os.environ.get('CK_MLPERF_DASHBOARD_DIR', '')

if cache_dir == '':
    import ck.kernel as ck
    repo_uoa = 'ck-mlperf'
    module_uoa = 'module'
    data_uoa = 'mlperf.inference'
    r = ck.access({'action':'find', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)
    cache_dir = r['path']

cache_path = os.path.join(cache_dir, cache_file)


# ### Divisions

# In[ ]:


divisions = [ 'closed', 'open' ]


# ### Maps for DataFrame construction

# In[ ]:


# Lowercase or camelcase or camelcase with space to camelcase.
scenario_to_str = {
    # SingleStream.
    'singlestream'  : 'SingleStream',
    'SingleStream'  : 'SingleStream',
    'Single Stream' : 'SingleStream',
    # MultiStream.
    'multistream'   : 'MultiStream',
    'MultiStream'   : 'MultiStream',
    'Multi Stream'  : 'MultiStream',
    # Server.
    'server'        : 'Server',
    'Server'        : 'Server',
    # Offline.
    'offline'       : 'Offline',
    'Offline'       : 'Offline',
}


# In[ ]:


division_to_str = {
    # Open.
    'open'   : 'Open',
    'Open'   : 'Open',
    # Closed.
    'closed' : 'Closed',
    'Closed' : 'Closed'
}


# In[ ]:


# dividiti-specific.
system_id_to_processor = {
    'firefly'   : 'Rockchip RK3399',
    'hikey960'  : 'HiSilicon Kirin960',
    'mate10pro' : 'HiSilicon Kirin970',
    'rpi4'      : 'Broadcom BCM2711B0',
}


# In[ ]:


accelerator_name_to_accelerator = {
    'NVIDIA Tesla T4': 'NVIDIA Tesla T4',
    'Nvidia Tesla T4': 'NVIDIA Tesla T4',
    'Tesla T4': 'NVIDIA Tesla T4',
    'Nvidia Tesla V100 SXM3': 'NVIDIA Tesla V100 SXM3',
    'tpu-v3.8': 'Google TPU v3-8', # NB: 8 TPU v3?
    'HanGuang 800': 'Alibaba HanGuang 800',
    'Goya': 'Habana Goya',
}


# ### Metrics for DataFrame construction

# In[ ]:


# Performance metrics: Stream in ms; MultiStream in #streams; Server in QPS; Offline in inputs/s).
performance_columns = [
    'P_{}_{}'.format(task, scenario)
    for task in ['IC1','IC2','OD1','OD2','NMT']
    for scenario in ['SS','MS','S','O']
]
# Accuracy metrics: Image Classification in Top1, %; Object Detection in mAP, %; Machine Translation in BLUE.
accuracy_columns = [
    'A_{}_{}'.format(task, scenario)
    for task in ['IC1','IC2','OD1','OD2','NMT']
    for scenario in ['SS','MS','S','O']
]
# Score columns.
score_columns = performance_columns + accuracy_columns


# ### Non-imagenet benchmarks

# In[ ]:


non_imagenet_benchmarks = {
    # Non-ImageNet benchmarks from the closed division.
    'ssd-small': {
        "name"  : "SSD-MobileNet-v1",
        "width" : 300,
        "height": 300,
    },
    'ssd-large': {
        "name"  : "SSD-ResNet34",
        "width" : 1200,
        "height": 1200,
    },
    'gnmt' : {
        "name"  : "GNMT",
        "width" : -1,
        "height": -1,
    },
    # Non-ImageNet benchmarks from the open division.
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
        "name" : "SSD-MobileNet-v1",
        "url" : "https://zenodo.org/record/3361502/files/ssd_mobilenet_v1_coco_2018_01_28.tar.gz",
        "width" : 300,
        "height" : 300,
        "provenance" : "Google",
    },
    'ssd-mobilenet-v1-non-quantized-mlperf' : {
        "name" : "SSD-MobileNet-v1 quantized",
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


# ## Code

# In[ ]:


# We use two modes: the 'spreadsheet' mode tries to mimic the official submission table as much as possible;
# the 'dashboard' mode uses a more appropriate layout for the CK dashboard.
def get_data(results_path=results_path, mode='spreadsheet'):
    dfs = []
    # FOR EACH division.
    for division in divisions:
        #if division == 'open': continue # skip
        # FOR EACH submitter.
        submitters_dir = os.path.join(results_path, division)
        submitters = [ fn for fn in os.listdir(submitters_dir) if os.path.isdir(os.path.join(submitters_dir, fn)) ]
        for submitter in submitters:
            # Selectively filter out submitters.
            #all_submitters_closed = [ 'Alibaba', 'CentaurTechnology', 'DellEMC', 'dividiti', 'FuriosaAI', 'Google', 'Habana', 'Hailo', 'Intel', 'NVIDIA', 'Qualcomm', 'Tencent' ]
            #if division == 'closed' and submitter not in all_submitters_closed: continue
            #all_submitters_open = [ 'dividiti', 'Habana', 'Inspur', 'NVIDIA', 'Qualcomm' ]
            #if division == 'open' and submitter not in all_submitters_open: continue
            # FOR EACH system.
            results_dir = os.path.join(submitters_dir, submitter, 'results')
            measurements_dir = os.path.join(submitters_dir, submitter, 'measurements')
            systems = [ fn for fn in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, fn)) ]
            for system in systems:
                system_dir = os.path.join(results_dir, system)
                system_json_name = system + '.json'
                system_json_path = os.path.join(submitters_dir, submitter, 'systems', system_json_name)
                with open(system_json_path) as system_json_file:
                    system_json = json.load(system_json_file)

                # Category.
                if system_json['status'] in [ 'available', 'Available' ]:
                    category = 'Available'
                elif system_json['status'] in [ 'preview', 'Preview' ]:
                    category = 'Preview'
                elif system_json['status'] in [ 'rdi', 'RDI', 'rdo', 'RDO' ]:
                    category = 'Research, Development, Other'
                elif system_json['status'] in [ 'Unofficial', 'unofficial' ]:
                    category = 'Unofficial'
                else:
                    raise Exception("Unsupported category '%s'!" % (system_json['status']))

                # System details.
                system_name = system_json['system_name']
                system_list = system.split('-')
                system_id = system_list[0]

                # Processor (CPU).
                processor = system_id_to_processor.get(system_id, system_json.get('host_processor_model_name', 'N/A'))
                processor_num = int(system_json.get('host_processors_per_node', 0))

                # Accelerator.
                # Tencent: https://github.com/mlperf/submissions_inference_0_5/issues/285
                accelerator_name = system_json.get('accelerator_model_name', 'N/A')
                accelerator_num = int(system_json.get('accelerators_per_node', 0))
                accelerator = accelerator_name_to_accelerator.get(accelerator_name, accelerator_name)

                # Software (framework).
                software = system_json['framework']

                # Default form factors and notes.
                # NB: Using space rather than empty string turns out to be better for dashboard.
                ff_m = ff_d = ff_s = ff_e = ' '
                notes = [' ']

                # Submitter-specific form factors and notes.
                submitter_str = submitter
                if submitter == 'dividiti':
                    # Form factors.
                    if system_id in [ 'hikey960', 'firefly', 'rpi4' ]: ff_e = 'x'
                    if system_id in [ 'mate10pro', 'hikey960' ]: ff_m = 'x'
                    if system_id in [ 'velociti' ]: ff_d = 'x'
                    # Notes.
                    if system_id == 'hikey960':
                        notes = 'Mobile chip in embedded form factor (development board).'
                    if division == 'open':
                        # Object Detection is collaboration between dividiti and Politecnico di Milano.
                        if system_id == 'velociti': submitter_str = 'dividiti + PoliMi'
                        if system == 'velociti-tensorflow-v1.14-cpu':
                            notes = 'In the Other category, since this Intel CPU is no longer available (end-of-life).'
                elif submitter == 'Alibaba':
                    ff_s = 'x'
                    if system_id == 'alibaba_cloud_t4':
                        notes = 'ECC off'
                elif submitter == 'DellEMC':
                    ff_s = 'x'
                    if system_id == 'R740_T4x4_tensorrt':
                        notes = 'ECC off'
                elif submitter == 'Google':
                    ff_s = 'x'
                    system_name = '{:d}x Cloud {:s}'.format(int(accelerator_num/8), accelerator)
                elif submitter == 'Habana':
                    ff_d = ff_s = ff_e = 'x'
                    if division == 'open':
                        if system_id == 'Goya_fast_latency':
                            notes = 'Low latency results ...'
                        if system_id == 'Goya_med_latency':
                            notes = 'Medium latency results ...'
                elif submitter == 'Intel':
                    if system_id == 'ICL':
                        ff_m = 'x'
                    else:
                        ff_s = 'x'
                elif submitter == 'NVIDIA':
                    if system_id == 'Xavier':
                        ff_e = 'x'
                        if division == 'closed':
                            notes = 'GPU and both DLAs are used in Offline and MultiStream'
                    elif system_id == 'TitanRTXx4':
                        ff_e = ff_s = ff_d = 'x'
                    elif system_id == 'T4x8':
                        ff_e = ff_s = 'x'
                    elif system_id == 'T4x20':
                        ff_s = 'x'
                    else:
                        raise Exception("Unsupported NVIDIA system '%s'!" % system_id)
                elif submitter == 'Qualcomm':
                    ff_m = 'x'
                    if division == 'open':
                        notes = 'Median latency. MultiStream: Both Hexagon Vector Extensions (HVX) and Hexagon Tensor Accelerator (HTA).'
                    if division == 'closed':
                        notes = 'Hexagon Vector Extensions being used.'
                elif submitter == 'Tencent':
                    ff_s = 'x'
                # Preview only.
                elif submitter == 'CentaurTechnology':
                    ff_d = ff_s = ff_e = 'x'
                elif submitter == 'Hailo':
                    ff_d = ff_e = 'x'
                # RDO only.
                elif submitter == 'FuriosaAI':
                    ff_d = ff_s = ff_e = 'x'
                # Open only.
                elif submitter == 'Inspur':
                    ff_s = 'x'
                else:
                    raise Exception("Unsupported division/submitter combination '%s'/'%s'!" % (division, submitter))

                details = 'https://github.com/mlperf/inference_results_v0.5/blob/master/{}/{}/systems/{}'.format(division, submitter, system_json_name)
                code    = 'https://github.com/mlperf/inference_results_v0.5/tree/master/{}/{}/code'.format(division, submitter)
                if category in [ 'Unofficial' ]:
                    details = 'N/A'
                    code    = 'N/A'

                # Create DataFrame for each row of the final table based on the division, submitter and system.
                data = [{
                    #
                    'ID'            : '-', # TODO: Fill in later.
                    'Submitter'     : submitter_str,
                    'System'        : system_name,
                    'Benchmark'     : '-', # TODO: Fill in later.
                    # Processor.
                    'Processor'     : processor,
                    'Processor #'   : processor_num,
                    # Accelerator.
                    'Accelerator'   : accelerator,
                    'Accelerator #' : accelerator_num if accelerator_num != '0' else '',
                    # Software.
                    'Software' : software,
                    # Form factor.
                    'FF_M'     : ff_m,
                    'FF_D'     : ff_d,
                    'FF_S'     : ff_s,
                    'FF_E'     : ff_e,
                    # Details. Code. Notes.
                    'Details'  : details,
                    'Code'     : code,
                    'Notes'    : notes,
                    # Misc.
                    'Division' : division_to_str.get(division, division),
                    'Category' : category,
                    'Task'     : '-', # TODO: Fill in later.
                    'Scenario' : '-', # TODO: Fill in later.
                }]
                # NB: 'Accelerator #' is important to sort Google's submissions correctly (not lexicographically).
                index = [
                    'Division', 'Category', 'Submitter', 'Accelerator #', 'System', 'Software', 'Benchmark' #, 'Task', 'Scenario'
                ]
                # Reset all scores.
                if mode == 'spreadsheet':
                    data[0].update({ score : '' for score in score_columns })

                # FOR EACH benchmark.
                benchmarks = [ fn for fn in os.listdir(system_dir) if os.path.isdir(os.path.join(system_dir, fn)) ]
                for (benchmark, benchmark_idx) in zip(benchmarks, range(len(benchmarks))):
                    is_last_benchmark = (benchmark_idx == len(benchmarks) - 1)
                    # Tencent and Inspur use resnet50.
                    benchmark_name = 'resnet' if benchmark == 'resnet50' else benchmark
                    # Benchmark (with notes).
                    benchmark_dict = non_imagenet_benchmarks.get(benchmark_name)
                    if benchmark_dict:
                        width = benchmark_dict['width']
                        height = benchmark_dict['height']
                    else:
                        if benchmark_name.endswith('96'):
                            side = 96
                        elif benchmark_name.endswith('128'):
                            side = 128
                        elif benchmark_name.endswith('160'):
                            side = 160
                        elif benchmark_name.endswith('192'):
                            side = 192
                        else:
                            side = 224
                        width = side
                        height = side
                    if width != -1 and height != -1:
                        # Benchmark (width x height).
                        benchmark_with_notes = '{} ({}x{})'.format(benchmark_name, width, height)
                    else:
                        # GNMT.
                        benchmark_with_notes = benchmark_name
                    # TODO: Rename to 'Model used, if not Closed Division default' for Open.
                    data[0]['Benchmark'] = benchmark_with_notes

                    # FOR EACH scenario.
                    benchmark_dir = os.path.join(system_dir, benchmark)
                    scenarios = [ fn for fn in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, fn)) ]
                    for scenario in scenarios:
                        if mode != 'spreadsheet':
                            data[0].update({ score : '' for score in score_columns })
                        scenario_str = scenario_to_str.get(scenario,'')
                        if scenario_str not in [ 'SingleStream', 'MultiStream', 'Server', 'Offline' ]: continue
                        experiment_dir = os.path.join(benchmark_dir, scenario)

                        # Extract accuracy.
                        if submitter == 'Hailo' and benchmark == 'ssd-small':
                            # https://github.com/mlperf/submissions_inference_0_5/issues/287
                            task = 'OD'
                            accuracy = 21.920 # ssd-small/SingleStream/accuracy/results.json
                        else:
                            accuracy_dir = os.path.join(experiment_dir, 'accuracy')
                            with open(os.path.join(accuracy_dir, 'accuracy.txt'), 'r') as accuracy_file:
                                accuracy_txt = accuracy_file.readlines()
                                accuracy_line = accuracy_txt[-1]
                            if accuracy_line.startswith('mAP'):
                                task = 'OD'
                                match = re.match('mAP\=([\d\.]+)\%', accuracy_line)
                                accuracy = float(match.group(1))
                            elif accuracy_line.startswith('accuracy'):
                                task = 'IC'
                                match = re.match('accuracy=(.+)%, good=(\d+), total=(\d+)', accuracy_line)
                                accuracy = float(match.group(1))
                            elif accuracy_line.startswith('BLEU'):
                                task = 'MT'
                                match = re.match('BLEU:\s*(.+)', accuracy_line)
                                accuracy = float(match.group(1))
                            else:
                                pprint(accuracy_txt)
                                raise Exception('Failed to extract accuracy information from "%s"' % accuracy_line)
                        data[0]['Task'] = { 'IC': 'Image Classification', 'OD': 'Object Detection', 'MT': 'Machine Translation' }.get(task)
                        data[0]['Scenario'] = scenario_to_str.get(scenario, scenario)
                        if submitter == 'Tencent' and scenario_str in [ 'SingleStream', 'Offline' ]:
                            # https://github.com/mlperf/submissions_inference_0_5/issues/286
                            performance_dirs = [ os.path.join(experiment_dir, 'performance') ]
                            # FIXME: Code below assumes directory name, not absolute path.
                        else:
                            performance_dirs = [ fn for fn in os.listdir(os.path.join(experiment_dir, 'performance'))
                                                 if os.path.isdir(os.path.join(experiment_dir, 'performance', fn)) and fn.startswith('run_') ]
                        # FOR EACH performance run. (Iterates over 5 runs for Server and other repeated runs.)
                        for performance_dir in performance_dirs:
                            # Reset notes from previous updates.
                            notes = [' ']
                            # If exist, read generic notes for each scenario.
                            notes_path = os.path.join(measurements_dir, system, benchmark, scenario, 'NOTES.txt')
                            if os.path.isfile(notes_path):
                                with open(notes_path, 'r') as notes_file:
                                    notes = notes_file.readlines()
                                data[0].update({'Notes' : notes})
                            # If exist, append performance notes to notes.
                            performance_notes_path = os.path.join(experiment_dir, 'performance', '%s.txt' % performance_dir)
                            if os.path.isfile(performance_notes_path):
                                with open(performance_notes_path, 'r') as performance_notes_file:
                                    performance_notes = performance_notes_file.readlines()
                                notes.append(performance_notes)
                                data[0].update({'Notes' : notes})
                            # Parse performance stats from the summary file.
                            with open(os.path.join(experiment_dir, 'performance', performance_dir, 'mlperf_log_summary.txt'), 'r') as summary_file:
                                summary_txt = summary_file.readlines()
                                for line in summary_txt:
                                    if re.match("Scenario", line):
                                        # NB: LoadGen scenario strings have spaces between 'Single'/'Multi' and 'Stream'.
                                        loadgen_scenario = line.split(": ",1)[1].strip()
                                        loadgen_scenario_str = scenario_to_str[loadgen_scenario]
                                        if loadgen_scenario_str != scenario_str:
                                            raise Exception("Expected '%s', parsed '%s'!" % (scenario_str, loadgen_scenario_str ))
                                        continue
                                    if scenario_str == "SingleStream":
                                        if re.match("90th percentile latency", line):
                                            score = line.split(": ",1)[1].strip()
                                            continue
                                    if scenario_str == "MultiStream":
                                        if re.match("Result is", line):
                                            status = line.split(": ",1)[1].strip()
                                            notes.append('Result = {}'.format(status))
                                            data[0].update({'Notes' : notes})
                                        if re.match("Per-query latency", line):
                                            latency_type = 'per-query'
                                        if re.match("Per-sample latency", line):
                                            latency_type = 'per-sample'
                                        # Match e.g. "Min latency (ns)                : 67438100"
                                        metric_regex = "(?P<metric>[A-Z][a-z]{2,3}) latency \(ns\)(\s)*:(\s)*(?P<ns>\d+)"
                                        metric_match = re.match(metric_regex, line)
                                        if metric_match:
                                            metric = metric_match.group('metric')
                                            ns = int(metric_match.group('ns'))
                                            ms = float(ns) * 1e-6
                                            note = '{:s} {:s} latency = {:.01f} ms'.format(metric, latency_type, ms)
                                            notes.append(note)
                                            data[0].update({'Notes' : notes})
                                            continue
                                        # Match e.g. "50.00 percentile latency (ns)   : 864943800"
                                        pc_regex = "(?P<pc>\d\d).00 percentile latency \(ns\)(\s)*:(\s)*(?P<ns>\d+)"
                                        pc_match = re.match(pc_regex, line)
                                        if pc_match:
                                            pc = int(pc_match.group('pc'))
                                            ns = int(pc_match.group('ns'))
                                            ms = float(ns) * 1e-6
                                            note = '{:d}% {:s} latency = {:.01f} ms'.format(pc, latency_type, ms)
                                            notes.append(note)
                                            data[0].update({'Notes' : notes})
                                            continue
                                        if re.match("Samples per query", line):
                                            score = line.split(": ",1)[1].strip()
                                            continue
                                    if scenario_str == "Server":
                                        if re.match("Scheduled samples per second", line):
                                            score = line.split(": ",1)[1].strip()
                                            continue
                                    if scenario_str == "Offline":
                                        if re.match("Samples per second", line):
                                            score = line.split(": ",1)[1].strip()
                                            continue
                            if scenario_str == 'SingleStream':
                                time_ns = int(score)
                                time_ms = time_ns * 1e-6
                            elif scenario_str == 'MultiStream':
                                num_streams = int(score)
                            elif scenario_str == 'Server':
                                queries_per_second = float(score)
                            elif scenario_str == 'Offline':
                                samples_per_second = float(score)

                            # Tasks.
                            if mode == 'spreadsheet':
                                ic1 = (task=='IC' and benchmark.startswith('mobilenet'))
                                ic2 = (task=='IC' and benchmark.startswith('resnet'))
                                od1 = (task=='OD' and benchmark=='ssd-small')
                                od2 = (task=='OD' and (benchmark=='ssd-large' or system_id=='velociti'))
                                nmt = (task=='MT')
                            else:
                                ic1 = (task=='IC')
                                ic2 = False
                                od1 = (task=='OD')
                                od2 = False
                                nmt = (task=='MT')
                            if scenario_str == 'SingleStream':
                                performance_str = '{:.03f}'.format(time_ms)
                                accuracy_str    = '{:.03f}'.format(accuracy)
                                if ic1:
                                    data[0]['A_IC1_SS'] = accuracy_str
                                    data[0]['P_IC1_SS'] = performance_str
                                elif ic2:
                                    data[0]['A_IC2_SS'] = accuracy_str
                                    data[0]['P_IC2_SS'] = performance_str
                                elif od1:
                                    data[0]['A_OD1_SS'] = accuracy_str
                                    data[0]['P_OD1_SS'] = performance_str
                                elif od2:
                                    data[0]['A_OD2_SS'] = accuracy_str
                                    data[0]['P_OD2_SS'] = performance_str
                                elif nmt:
                                    data[0]['A_NMT_SS'] = accuracy_str
                                    data[0]['P_NMT_SS'] = performance_str
                            elif scenario_str == 'MultiStream':
                                performance_str = '{:d}'.format(num_streams)
                                accuracy_str    = '{:.03f}'.format(accuracy)
                                if ic1:
                                    data[0]['A_IC1_MS'] = accuracy_str
                                    data[0]['P_IC1_MS'] = performance_str
                                elif ic2:
                                    data[0]['A_IC2_MS'] = accuracy_str
                                    data[0]['P_IC2_MS'] = performance_str
                                elif od1:
                                    data[0]['A_OD1_MS'] = accuracy_str
                                    data[0]['P_OD1_MS'] = performance_str
                                elif od2:
                                    data[0]['A_OD2_MS'] = accuracy_str
                                    data[0]['P_OD2_MS'] = performance_str
                                elif nmt:
                                    data[0]['A_NMT_MS'] = accuracy_str
                                    data[0]['P_NMT_MS'] = performance_str
                            elif scenario_str == 'Server':
                                performance_str = '{:.03f}'.format(queries_per_second)
                                accuracy_str    = '{:.03f}'.format(accuracy)
                                if ic1:
                                    data[0]['A_IC1_S'] = accuracy_str
                                    data[0]['P_IC1_S'] = performance_str
                                elif ic2:
                                    data[0]['A_IC2_S'] = accuracy_str
                                    data[0]['P_IC2_S'] = performance_str
                                elif od1:
                                    data[0]['A_OD1_S'] = accuracy_str
                                    data[0]['P_OD1_S'] = performance_str
                                elif od2:
                                    data[0]['A_OD2_S'] = accuracy_str
                                    data[0]['P_OD2_S'] = performance_str
                                elif nmt:
                                    data[0]['A_NMT_S'] = accuracy_str
                                    data[0]['P_NMT_S'] = performance_str
                            elif scenario_str == 'Offline':
                                performance_str = '{:.03f}'.format(samples_per_second)
                                accuracy_str    = '{:.03f}'.format(accuracy)
                                if ic1:
                                    data[0]['A_IC1_O'] = accuracy_str
                                    data[0]['P_IC1_O'] = performance_str
                                elif ic2:
                                    data[0]['A_IC2_O'] = accuracy_str
                                    data[0]['P_IC2_O'] = performance_str
                                elif od1:
                                    data[0]['A_OD1_O'] = accuracy_str
                                    data[0]['P_OD1_O'] = performance_str
                                elif od2:
                                    data[0]['A_OD2_O'] = accuracy_str
                                    data[0]['P_OD2_O'] = performance_str
                                elif nmt:
                                    data[0]['A_NMT_O'] = accuracy_str
                                    data[0]['P_NMT_O'] = performance_str
                            else:
                                print('Skipping unsupported task/scenario combination!')
                                continue
                            if mode != 'spreadsheet':
                                df = pd.DataFrame(data)
                                df = df.set_index(index)
                                dfs.append(df)
                            # END OF FOR EACH performance run
                    # END OF FOR EACH scenario
                    if mode == 'spreadsheet':
                        # For closed, multiple benchmarks can share the same row, so the Benchmark field can be misleading.
                        if division == 'closed': data[0]['Benchmark'] = ''
                        if is_last_benchmark or (division == 'open' and submitter == 'dividiti'):
                            df = pd.DataFrame(data)
                            df = df.set_index(index)
                            dfs.append(df)
                    # For the spreadsheet mode, include multiple benchmarks per row.
                # END OF FOR EACH benchmark
            # END OF FOR EACH system
        # END OF FOR EACH submitter
    # END OF FOR EACH division

    # Concatenate all thus constructed DataFrames (i.e. stack on top of each other).
    df = pd.concat(dfs)
    # Temporarily capitalize the first letter in 'dividiti' for correct sorting and then back.
    df = df         .rename(index={'dividiti':'Dividiti', 'dividiti + PoliMi':'Dividiti + PoliMi'})         .sort_index(ascending=True)         .rename(index={'Dividiti':'dividiti', 'Dividiti + PoliMi':'dividiti + PoliMi'})
    # Reset the index, but keep Division and Category there.
    df = df.reset_index(level=index[2:])
    df['ID'] = [ 'Inf-0.5-{:03d}'.format(ID) for ID in range(1, len(df)+1) ]
    # Mimic the official template.
    columns = [ 'ID', 'Submitter', 'System', 'Benchmark' ]
    columns += score_columns
    columns += [ 'Processor', 'Processor #', 'Accelerator', 'Accelerator #', 'Software',
                'FF_M', 'FF_D', 'FF_S', 'FF_E', 'Details', 'Code', 'Notes' ]
    # Finalize the table.
    if mode == 'spreadsheet':
        df = df[columns]
    else:
        df = df.reset_index().set_index(keys=[ 'ID', 'Division', 'Category', 'Submitter', 'System', 'Benchmark' ], drop=False)
        df[score_columns] = df[score_columns].apply(pd.to_numeric).astype('float32')

    return df


# ## Dump the table for the CK dashboard

if os.path.exists(cache_path):
    print('Skipping existing dashboard file at \'{}\' ...'.format(cache_path))
    exit(1)
else:
    # Store the table in a simplified format.
    print('Storing dashboard file to \'{}\' ...'.format(cache_path))
    df = get_data(results_path=results_path, mode='dashboard')
    df.to_pickle(path=cache_path, protocol=cache_protocol, compression=cache_compression)
