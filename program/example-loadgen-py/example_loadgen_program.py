#!/usr/bin/env python3

# ck virtual env --tags=loadgen,lib,python-package --shell_cmd="loadgen_test_script.py --config_file /Users/lg4/CK-TOOLS/mlperf-inference-upstream.master/inference/v0.5/mlperf.conf"

import argparse
import array
import random
import time

import numpy as np
import mlperf_loadgen as lg

dataset_size    = 20
dataset         = [10*i for i in range(dataset_size)]
labelset        = [10*i+random.randint(0,1) for i in range(dataset_size)]


def predict_label(x_vector):
    time.sleep(0.030)   # fractional seconds
    return int(x_vector/10)+1


def issue_queries(query_samples):

    printable_query = [(qs.index, qs.id) for qs in query_samples]
    print("LG: issue_queries( {} )".format(printable_query))

    predicted_results = {}
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        x_vector        = dataset[query_index]
        predicted_label = predict_label(x_vector)

        predicted_results[query_index] = predicted_label
    print("LG: predicted_results = {}".format(predicted_results))

    response = []
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        response_array = array.array("B", np.array(predicted_results[query_index], np.float32).tobytes())
        bi = response_array.buffer_info()
        response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)


def flush_queries():
    print("LG called flush_queries()")

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


def load_query_samples(sample_indices):
    print("LG called load_query_samples({})".format(sample_indices))

def unload_query_samples(sample_indices):
    print("LG called unload_query_samples({})".format(sample_indices))
    print("")


def benchmark_using_loadgen( scenario_str, mode_str, samples_in_mem, config_filepath ):
    "Perform the benchmark using python API for the LoadGen librar"

    scenario = {
        'SingleStream':     lg.TestScenario.SingleStream,
        'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[scenario_str]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[mode_str]

    ts = lg.TestSettings()
    if(config_filepath):
        ts.FromConfig(config_filepath, 'random_model_name', scenario_str)
    ts.scenario = scenario
    ts.mode     = mode

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(dataset_size, samples_in_mem, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


def main():
    "Parse command line and feed the benchmark_using_loadgen function"

    arg_parser  = argparse.ArgumentParser()
    arg_parser.add_argument('--scenario',       type=str,   default='SingleStream',     help='LoadGen testing scenario')
    arg_parser.add_argument('--mode',           type=str,   default='AccuracyOnly',     help='LoadGen testing mode')
    arg_parser.add_argument('--samples_in_mem', type=int,   default=8,                  help='Num of samples memory can hold')
    arg_parser.add_argument('--config_file',    type=str,   default='',                 help='Path to LoadGen config file')
    args        = arg_parser.parse_args()

    benchmark_using_loadgen( args.scenario, args.mode, args.samples_in_mem, args.config_file )


main()