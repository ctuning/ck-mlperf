#!/usr/bin/env python3

import array
import numpy as np
import os
import sys
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools

import mlperf_loadgen as lg


## LoadGen test properties:
#
LOADGEN_SCENARIO        = os.getenv('CK_LOADGEN_SCENARIO', 'SingleStream')
LOADGEN_MODE            = os.getenv('CK_LOADGEN_MODE', 'AccuracyOnly')
LOADGEN_BUFFER_SIZE     = int(os.getenv('CK_LOADGEN_BUFFER_SIZE'))          # set to how many samples are you prepared to keep in memory at once
LOADGEN_DATASET_SIZE    = int(os.getenv('CK_LOADGEN_DATASET_SIZE'))         # set to how many total samples to choose from (0 = full set)
LOADGEN_CONF_FILE       = os.getenv('CK_LOADGEN_CONF_FILE', '')
LOADGEN_COUNT_OVERRIDE      = os.getenv('CK_LOADGEN_COUNT_OVERRIDE', '')    # if not set, use value from LoadGen's config file
LOADGEN_MULTISTREAMNESS = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')       # if not set, use value from LoadGen's config file
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', '1'))

## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')
MODEL_INPUT_DATA_TYPE   = os.getenv('ML_MODEL_INPUT_DATA_TYPE', 'float32')
MODEL_DATA_TYPE         = os.getenv('ML_MODEL_DATA_TYPE', '(unknown)')
MODEL_SOFTMAX_LAYER     = os.getenv('CK_ENV_ONNX_MODEL_OUTPUT_LAYER_NAME', os.getenv('CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME', ''))


## Internal processing:
#
INTERMEDIATE_DATA_TYPE  = np.float32    # default for internal conversion
#INTERMEDIATE_DATA_TYPE  = np.int8       # affects the accuracy a bit


## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.fromstring(GIVEN_CHANNEL_MEANS, dtype=np.float32, sep=' ').astype(INTERMEDIATE_DATA_TYPE)
    if MODEL_COLOURS_BGR:
        GIVEN_CHANNEL_MEANS = GIVEN_CHANNEL_MEANS[::-1]     # swapping Red and Blue colour channels

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
IMAGE_LIST_FILE         = os.path.join(IMAGE_DIR, os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF'))
IMAGE_DATA_TYPE         = np.dtype( os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DATA_TYPE', 'uint8') )

## Misc
#
VERBOSITY_LEVEL         = int(os.getenv('CK_VERBOSE', '0'))


# Load preprocessed image filepaths:
with open(IMAGE_LIST_FILE, 'r') as f:
    image_path_list = [ os.path.join(IMAGE_DIR, s.strip()) for s in f ]
LOADGEN_DATASET_SIZE = LOADGEN_DATASET_SIZE or len(image_path_list)


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def tick(letter, quantity=1):
    print(letter + (str(quantity) if quantity>1 else ''), end='')


# Currently loaded preprocessed images are stored in a dictionary:
preprocessed_image_buffer = {}


def load_query_samples(sample_indices):     # 0-based indices in our whole dataset
    global MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS

    print("load_query_samples({})".format(sample_indices))

    tick('B', len(sample_indices))

    for sample_index in sample_indices:
        img_filename = image_path_list[sample_index]
        img = np.fromfile(img_filename, IMAGE_DATA_TYPE)
        img = img.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3))
        if MODEL_COLOURS_BGR:
            img = img[...,::-1]     # swapping Red and Blue colour channels

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(np.float32)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img, axis=(0,1), keepdims=True)

        if MODEL_INPUT_DATA_TYPE == 'int8' or INTERMEDIATE_DATA_TYPE==np.int8:
            img = np.clip(img, -128, 127).astype(INTERMEDIATE_DATA_TYPE)

        nhwc_img = img if MODEL_DATA_LAYOUT == 'NHWC' else img.transpose(2,0,1)

        preprocessed_image_buffer[sample_index] = np.array(nhwc_img).ravel().astype(MODEL_INPUT_DATA_TYPE)
        tick('l')
    print('')


def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    preprocessed_image_buffer = {}
    tick('U')
    print('')


def initialize_predictor():
    global pycuda_context
    global d_inputs, h_d_outputs, h_output, model_bindings, cuda_stream
    global num_labels, model_classes
    global trt_context
    global MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS
    global BATCH_SIZE
    global max_batch_size

    # Load the TensorRT model from file
    pycuda_context = pycuda.tools.make_default_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    try:
        with open(MODEL_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            print('[TRT] successfully loaded')
    except:
        pycuda_context.pop()
        raise RuntimeError('TensorRT model file {} is not found or corrupted'.format(MODEL_PATH))

    max_batch_size      = trt_engine.max_batch_size

    if BATCH_SIZE>max_batch_size:
        pycuda_context.pop()
        raise RuntimeError("Desired batch_size ({}) exceeds max_batch_size of the model ({})".format(BATCH_SIZE,max_batch_size))

    d_inputs, h_d_outputs, model_bindings = [], [], []
    for interface_layer in trt_engine:
        dtype   = trt_engine.get_binding_dtype(interface_layer)
        shape   = trt_engine.get_binding_shape(interface_layer)
        size    = trt.volume(shape) * max_batch_size

        dev_mem = cuda.mem_alloc(size * dtype.itemsize)
        model_bindings.append( int(dev_mem) )

        if trt_engine.binding_is_input(interface_layer):
            interface_type = 'Input'
            d_inputs.append(dev_mem)
            model_input_shape   = shape
        else:
            interface_type = 'Output'
            host_mem    = cuda.pagelocked_empty(size, trt.nptype(dtype))
            h_d_outputs.append({ 'host_mem': host_mem, 'dev_mem': dev_mem })
            if MODEL_SOFTMAX_LAYER=='' or interface_layer == MODEL_SOFTMAX_LAYER:
                model_output_shape  = shape
                h_output            = host_mem

        print("{} layer {}: dtype={}, shape={}, elements_per_max_batch={}".format(interface_type, interface_layer, dtype, shape, size))

    cuda_stream     = cuda.Stream()
    num_labels      = len(load_labels(LABELS_PATH))
    model_classes   = trt.volume(model_output_shape)

    if MODEL_DATA_LAYOUT == 'NHWC':
        (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS) = model_input_shape
    else:
        (MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH) = model_input_shape

    trt_context   = trt_engine.create_execution_context()


def predict_labels_for_batch(batch_data):
    global d_inputs, h_d_outputs, h_output, model_bindings, cuda_stream
    global num_labels, model_classes
    global trt_context
    global max_batch_size

    batch_size          = len(batch_data)
    flat_float_batch    = np.ravel(batch_data)

    begin_time = time.time()

    cuda.memcpy_htod_async(d_inputs[0], flat_float_batch, cuda_stream)  # assuming one input layer for image classification
    trt_context.execute_async(bindings=model_bindings, batch_size=batch_size, stream_handle=cuda_stream.handle)
    for output in h_d_outputs:
        cuda.memcpy_dtoh_async(output['host_mem'], output['dev_mem'], cuda_stream)
    cuda_stream.synchronize()

    classification_time = time.time() - begin_time

    print("[batch of {}] inference={:.2f} ms".format(batch_size, classification_time*1000))

    batch_results           = np.split(h_output, max_batch_size)  # where each row is a softmax_vector for one sample

    if model_classes==1:
        batch_predicted_labels  = batch_results[:batch_size]
    else:
        batch_predicted_labels  = [ np.argmax(batch_results[k][-num_labels:]) for k in range(batch_size) ]

    return batch_predicted_labels


def issue_queries(query_samples):

    global BATCH_SIZE

    if VERBOSITY_LEVEL:
        printable_query = [(qs.index, qs.id) for qs in query_samples]
        print("issue_queries( {} )".format(printable_query))
    tick('Q', len(query_samples))

    for j in range(0, len(query_samples), BATCH_SIZE):
        batch       = query_samples[j:j+BATCH_SIZE]   # NB: the last one may be shorter than BATCH_SIZE in length
        batch_data  = [ preprocessed_image_buffer[qs.index] for qs in batch ]

        batch_predicted_labels = predict_labels_for_batch(batch_data)
        tick('p', len(batch))
        if VERBOSITY_LEVEL:
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
    print("LG called process_latencies({})".format(latencies_ns))

    latencies_size      = len(latencies_ns)
    latencies_avg       = int(sum(latencies_ns)/latencies_size)
    latencies_sorted    = sorted(latencies_ns)
    latencies_p50       = int(latencies_size * 0.5);
    latencies_p90       = int(latencies_size * 0.9);

    print("--------------------------------------------------------------------")
    print("|                LATENCIES (in nanoseconds and fps)                |")
    print("--------------------------------------------------------------------")
    print("Number of queries run:       {:9d}".format(latencies_size))
    print("Min latency:                 {:9d} ns   ({:.3f} fps)".format(latencies_sorted[0], 1e9/latencies_sorted[0]))
    print("Median latency:              {:9d} ns   ({:.3f} fps)".format(latencies_sorted[latencies_p50], 1e9/latencies_sorted[latencies_p50]))
    print("Average latency:             {:9d} ns   ({:.3f} fps)".format(latencies_avg, 1e9/latencies_avg))
    print("90 percentile latency:       {:9d} ns   ({:.3f} fps)".format(latencies_sorted[latencies_p90], 1e9/latencies_sorted[latencies_p90]))
    print("Max latency:                 {:9d} ns   ({:.3f} fps)".format(latencies_sorted[-1], 1e9/latencies_sorted[-1]))
    print("--------------------------------------------------------------------")


def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    global pycuda_context
    initialize_predictor()

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
    if LOADGEN_CONF_FILE:
        ts.FromConfig(LOADGEN_CONF_FILE, 'random_model_name', LOADGEN_SCENARIO)
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
    pycuda_context.pop()


benchmark_using_loadgen()
