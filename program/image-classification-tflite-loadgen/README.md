# MLPerf Inference - Image Classification - TFLite with LoadGen

Running this program is similar to running [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite),
as described in the [MLPerf Inference repo](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection/optional_harness_ck/classification/tflite).

## Detect LoadGen Config file
```
$ ck detect soft --tags=config,loadgen,image-classification-tflite
```

## Run once
```
firefly $ ck benchmark program:image-classification-tflite-loadgen \
--speed --repetitions=1 \
--env.CK_VERBOSE=1 \
--env.CK_LOADGEN_SCENARIO=SingleStream \
--env.CK_LOADGEN_MODE=PerformanceOnly \
--env.CK_LOADGEN_DATASET_SIZE=1024 \
--env.CK_LOADGEN_BUFFER_SIZE=1024 \
--dep_add_tags.weights=model,tflite,resnet \
--dep_add_tags.library=tflite,v1.15 \
--dep_add_tags.compiler=gcc,v7 \
--dep_add_tags.images=side.224,preprocessed \
--dep_add_tags.loadgen-config-file=image-classification-tflite \
--dep_add_tags.python=v3 \
--skip_print_timers
...
------------------------------------------------------------
|            LATENCIES (in nanoseconds and fps)            |
------------------------------------------------------------
Number of queries run: 1024
Min latency:                      397952762ns  (2.51286 fps)
Median latency:                   426440993ns  (2.34499 fps)
Average latency:                  433287227ns  (2.30794 fps)
90 percentile latency:            460194271ns  (2.173 fps)
Max latency:                      679467557ns  (1.47174 fps)
------------------------------------------------------------ 
```

## Explore different models

See `run.sh` scripts we used to generate the [MLPerf Inference v0.5 results](https://mlperf.org/inference-results/) for more details:
```
$ ck list ck-mlperf:script:mlperf-inference-v0.5.*.image-classification
mlperf-inference-v0.5.closed.image-classification
mlperf-inference-v0.5.open.image-classification
```
