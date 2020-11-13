#!/usr/bin/env python3

import array
import os
import sys
import time
import numpy as np
import torch

from imagenet_helper import (load_image_by_index_and_normalize, image_list, BATCH_SIZE,
    MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_INPUT_DATA_TYPE)

import mlperf_loadgen as lg


## LoadGen test properties:
#
LOADGEN_SCENARIO        = os.getenv('CK_LOADGEN_SCENARIO', 'SingleStream')
LOADGEN_MODE            = os.getenv('CK_LOADGEN_MODE', 'AccuracyOnly')
LOADGEN_BUFFER_SIZE     = int(os.getenv('CK_LOADGEN_BUFFER_SIZE'))          # set to how many samples are you prepared to keep in memory at once
LOADGEN_DATASET_SIZE    = int(os.getenv('CK_LOADGEN_DATASET_SIZE'))         # set to how many total samples to choose from (0 = full set)
LOADGEN_COUNT_OVERRIDE  = os.getenv('CK_LOADGEN_COUNT_OVERRIDE', '')        # if not set, use value from LoadGen's config file
LOADGEN_MULTISTREAMNESS = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')       # if not set, use value from LoadGen's config file

MLPERF_CONF_PATH        = os.environ['CK_ENV_MLPERF_INFERENCE_MLPERF_CONF']
USER_CONF_PATH          = os.environ['CK_LOADGEN_USER_CONF']


## Model properties:
#
MODEL_NAME              = os.getenv('ML_MODEL_MODEL_NAME', 'resnet50')

## Enabling GPU if available and not disabled:
#
USE_CUDA                = torch.cuda.is_available() and (os.getenv('CK_DISABLE_CUDA', '0') in ('NO', 'no', 'OFF', 'off', '0'))


## Misc
#
VERBOSITY_LEVEL         = int(os.getenv('CK_VERBOSE', '0'))


# Load preprocessed image filepaths:
LOADGEN_DATASET_SIZE = LOADGEN_DATASET_SIZE or len(image_list)


def tick(letter, quantity=1):
    if VERBOSITY_LEVEL:
        print(letter + (str(quantity) if quantity>1 else ''), end='')


# Currently loaded preprocessed images are stored in pre-allocated numpy arrays:
preprocessed_image_buffer = None
preprocessed_image_map = np.empty(LOADGEN_DATASET_SIZE, dtype=np.int)   # this type should be able to hold indices in range 0:LOADGEN_DATASET_SIZE


def load_query_samples(sample_indices):     # 0-based indices in our whole dataset
    global preprocessed_image_buffer

    if VERBOSITY_LEVEL > 1:
        print("load_query_samples({})".format(sample_indices))

    len_sample_indices = len(sample_indices)

    tick('B', len_sample_indices)

    if preprocessed_image_buffer is None:     # only do this once, once we know the expected size of the buffer
        preprocessed_image_buffer = np.empty((len_sample_indices, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), dtype=MODEL_INPUT_DATA_TYPE)

    for buffer_index, sample_index in zip(range(len_sample_indices), sample_indices):
        preprocessed_image_map[sample_index] = buffer_index
        preprocessed_image_buffer[buffer_index] = np.array( load_image_by_index_and_normalize(sample_index) )

        tick('l')

    if VERBOSITY_LEVEL:
        print('')


def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    tick('U')

    if VERBOSITY_LEVEL:
        print('')


def issue_queries(query_samples):

    global BATCH_SIZE
    global model

    if VERBOSITY_LEVEL > 2:
        printable_query = [(qs.index, qs.id) for qs in query_samples]
        print("issue_queries( {} )".format(printable_query))
    tick('Q', len(query_samples))

    for j in range(0, len(query_samples), BATCH_SIZE):
        batch       = query_samples[j:j+BATCH_SIZE]   # NB: the last one may be shorter than BATCH_SIZE in length
        batch_data  = preprocessed_image_buffer[preprocessed_image_map[ [qs.index for qs in batch] ]]
        torch_batch = torch.from_numpy( batch_data )

        begin_time = time.time()

        # move the input to GPU for speed if available
        if USE_CUDA:
            torch_batch = torch_batch.to('cuda')

        with torch.no_grad():
            trimmed_batch_results = model( torch_batch )

        inference_time_s = time.time() - begin_time

        actual_batch_size = len(trimmed_batch_results)

        if VERBOSITY_LEVEL > 1:
            print("[batch of {}] inference={:.2f} ms".format(actual_batch_size, inference_time_s*1000))

        batch_predicted_labels  = torch.argmax(trimmed_batch_results, dim=1).tolist()

        tick('p', len(batch))
        if VERBOSITY_LEVEL > 2:
            print("predicted_batch_results = {}".format(batch_predicted_labels))

        response = []
        response_array_refs = []    # This is needed to guarantee that the individual buffers to which we keep extra-Pythonian references, do not get garbage-collected.
        for qs, predicted_label in zip(batch, batch_predicted_labels):

            response_array = array.array("B", np.array(predicted_label, np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        #tick('R', len(response))
    sys.stdout.flush()


def flush_queries():
    pass


def process_latencies(latencies_ns):
    latencies_ms = [ (ns * 1e-6) for ns in latencies_ns ]
    print("LG called process_latencies({})".format(latencies_ms))

    latencies_size      = len(latencies_ms)
    latencies_avg       = int(sum(latencies_ms)/latencies_size)
    latencies_sorted    = sorted(latencies_ms)
    latencies_p50       = int(latencies_size * 0.5);
    latencies_p90       = int(latencies_size * 0.9);
    latencies_p99       = int(latencies_size * 0.99);

    print("--------------------------------------------------------------------")
    print("|                LATENCIES (in milliseconds and fps)               |")
    print("--------------------------------------------------------------------")
    print("Number of samples run:       {:9d}".format(latencies_size))
    print("Min latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[0], 1e3/latencies_sorted[0]))
    print("Median latency:              {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p50], 1e3/latencies_sorted[latencies_p50]))
    print("Average latency:             {:9.2f} ms   ({:.3f} fps)".format(latencies_avg, 1e3/latencies_avg))
    print("90 percentile latency:       {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p90], 1e3/latencies_sorted[latencies_p90]))
    print("99 percentile latency:       {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[latencies_p99], 1e3/latencies_sorted[latencies_p99]))
    print("Max latency:                 {:9.2f} ms   ({:.3f} fps)".format(latencies_sorted[-1], 1e3/latencies_sorted[-1]))
    print("--------------------------------------------------------------------")


def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    global model

    # Load the [cached] Torch model
    torchvision_version = ''    # master by default
    try:
        import torchvision
        torchvision_version = ':v' + torchvision.__version__
    except Exception:
        pass

    model = torch.hub.load('pytorch/vision' + torchvision_version, MODEL_NAME, pretrained=True)
    model.eval()

    # move the model to GPU for speed if available
    if USE_CUDA:
        model.to('cuda')

    scenario = {
        'SingleStream':     lg.TestScenario.SingleStream,
        'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[LOADGEN_SCENARIO]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[LOADGEN_MODE]

    ts = lg.TestSettings()
    ts.FromConfig(MLPERF_CONF_PATH, MODEL_NAME, LOADGEN_SCENARIO)
    ts.FromConfig(USER_CONF_PATH, MODEL_NAME, LOADGEN_SCENARIO)
    ts.scenario = scenario
    ts.mode     = mode

    if LOADGEN_MULTISTREAMNESS:
        ts.multi_stream_samples_per_query = int(LOADGEN_MULTISTREAMNESS)

    if LOADGEN_COUNT_OVERRIDE:
        ts.min_query_count = int(LOADGEN_COUNT_OVERRIDE)
        ts.max_query_count = int(LOADGEN_COUNT_OVERRIDE)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(LOADGEN_DATASET_SIZE, LOADGEN_BUFFER_SIZE, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


try:
    benchmark_using_loadgen()
except Exception as e:
    print('{}'.format(e))
