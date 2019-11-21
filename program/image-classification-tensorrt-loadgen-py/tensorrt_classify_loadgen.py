#!/usr/bin/env python3

import array
import numpy as np
import os

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
LOADGEN_MULTISTREAMNESS = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')       # if not set, use value from LoadGen's config file
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', '1'))

## Model properties:
#

MODEL_PATH              = os.environ['CK_ENV_TENSORRT_MODEL_FILENAME']
MODEL_DATA_LAYOUT       = os.getenv('ML_MODEL_DATA_LAYOUT', 'NCHW')
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']
MODEL_COLOURS_BGR       = os.getenv('ML_MODEL_COLOUR_CHANNELS_BGR', 'NO') in ('YES', 'yes', 'ON', 'on', '1')

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv('ML_MODEL_NORMALIZE_DATA') in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv('ML_MODEL_SUBTRACT_MEAN', 'YES') in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv('ML_MODEL_GIVEN_CHANNEL_MEANS', '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.array(GIVEN_CHANNEL_MEANS.split(' '), dtype=np.float32)
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


# Currently loaded preprocessed images are stored in a dictionary:
preprocessed_image_buffer = {}


def load_query_samples(sample_indices):     # 0-based indices in our whole dataset
    global MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS

    print("load_query_samples({})".format(sample_indices))

    print('B', end='')

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
                    img -= np.mean(img)

        nhwc_img = img if MODEL_DATA_LAYOUT == 'NHWC' else img.transpose(2,0,1)

        preprocessed_image_buffer[sample_index] = np.array(nhwc_img).ravel().astype(np.float32)
        print('l', end='')
    print('')


def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    preprocessed_image_buffer = {}
    print('U')


def initialize_predictor():
    global default_context
    global h_input, h_output, d_input, d_output, cuda_stream
    global bg_class_offset
    global execution_context
    global MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS
    global BATCH_SIZE
    global max_batch_size

    # Load the TensorRT model from file
    default_context = pycuda.tools.make_default_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    try:
        with open(MODEL_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            print('[TRT] successfully loaded')
    except:
        print('[TRT] file {} is not found or corrupted'.format(MODEL_PATH))
        raise

    model_input_shape   = trt_engine.get_binding_shape(0)
    model_output_shape  = trt_engine.get_binding_shape(1)
    max_batch_size      = trt_engine.max_batch_size

    if BATCH_SIZE>max_batch_size:
        default_context.pop()
        raise RuntimeError("Desired batch_size ({}) exceeds max_batch_size of the model ({})".format(BATCH_SIZE,max_batch_size))

    IMAGE_DATATYPE      = np.float32
    h_input             = cuda.pagelocked_empty(max_batch_size*trt.volume(model_input_shape), dtype=IMAGE_DATATYPE)
    h_output            = cuda.pagelocked_empty(max_batch_size*trt.volume(model_output_shape), dtype=IMAGE_DATATYPE)
    print('Allocated device memory buffers: input_size={} output_size={}'.format(h_input.nbytes, h_output.nbytes))
    d_input             = cuda.mem_alloc(h_input.nbytes)
    d_output            = cuda.mem_alloc(h_output.nbytes)
    cuda_stream         = cuda.Stream()

    model_classes       = model_output_shape[0]
    labels              = load_labels(LABELS_PATH)
    bg_class_offset     = model_classes-len(labels)  # 1 means the labels represent classes 1..1000 and the background class 0 has to be skipped

    if MODEL_DATA_LAYOUT == 'NHWC':
        (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, MODEL_IMAGE_CHANNELS) = model_input_shape
    else:
        (MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH) = model_input_shape

    execution_context   = trt_engine.create_execution_context()


def predict_labels_for_batch(batch_image_map):
    global default_context
    global h_input, h_output, d_input, d_output, cuda_stream
    global bg_class_offset
    global execution_context
    global max_batch_size

    items_this_batch    = len(batch_image_map)
    query_indices       = list(batch_image_map.keys())
    h_input             = np.ravel(list(batch_image_map.values()))

    cuda.memcpy_htod_async(d_input, h_input, cuda_stream)
    execution_context.execute_async(bindings=[int(d_input), int(d_output)], batch_size=items_this_batch, stream_handle=cuda_stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, cuda_stream)
    cuda_stream.synchronize()

    softmax_matrix      = np.split(h_output, max_batch_size)  # where each row is a softmax_vector for one sample
    batch_argmax_map    = {}
    for k in range(items_this_batch):
        softmax_vector = softmax_matrix[k][bg_class_offset:]
        batch_argmax_map[query_indices[k]] = np.argmax(softmax_vector)

    return batch_argmax_map


def issue_queries(query_samples):

    global BATCH_SIZE

    printable_query = [(qs.index, qs.id) for qs in query_samples]
    if VERBOSITY_LEVEL:
        print("issue_queries( {} )".format(printable_query))
    print('Q'+(str(len(query_samples)) if len(query_samples)>1 else ''), end='')

    response = []
    response_array_refs = []    # This is needed to guarantee that the individual buffers to which we keep extra-Pythonian references, do not get garbage-collected.
    for j in range(0, len(query_samples), BATCH_SIZE):
        batch = query_samples[j:j+BATCH_SIZE]   # NB: the last one may be shorter than BATCH_SIZE in length
        batch_image_map = {}
        for qs in batch:
            query_index, query_id = qs.index, qs.id
            batch_image_map[query_index] = preprocessed_image_buffer[query_index]

        predicted_batch_results = predict_labels_for_batch(batch_image_map)
        print('p'+(str(len(batch)) if BATCH_SIZE>1 else ''), end='')
        if VERBOSITY_LEVEL:
            print("predicted_batch_results = {}".format(predicted_batch_results))

        for qs in batch:
            query_index, query_id = qs.index, qs.id

            response_array = array.array("B", np.array(predicted_batch_results[query_index], np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)


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

    global default_context
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

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(LOADGEN_DATASET_SIZE, LOADGEN_BUFFER_SIZE, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
    default_context.pop()


benchmark_using_loadgen()
