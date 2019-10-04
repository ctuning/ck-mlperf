#!/bin/bash

division=closed
task=image-classification

system=rpi4
library=tflite,v1.15
if [ ${system} = 'rpi4' ]
then
  compiler=gcc,v8
else
  compiler=gcc,v7
fi

scenario=SingleStream
scenario_tag=singlestream
mode=AccuracyOnly
mode_tag=accuracy
dataset_size=5000
buffer_size=500

# Image classification models (in the closed division).
models=( 'mobilenet' 'resnet' )
# Preprocessing methods.
methods=( 'preprocessing,using-opencv' 'preprocessing,using-tensorflow' )
for i in $(seq 1 ${#models[@]}); do
  model=${models[${i}-1]}
  method=${methods[0]} # use the same preprocessing for any model
  echo "Benchmarking '${model}' ..."
  echo ck benchmark program:image-classification-tflite-loadgen \
  --speed --repetitions=1 \
  --env.CK_LOADGEN_SCENARIO=${scenario} \
  --env.CK_LOADGEN_MODE=${mode} \
  --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
  --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
  --env.CK_VERBOSE=2 \
  --dep_add_tags.weights=${model} \
  --dep_add_tags.images=${method} \
  --dep_add_tags.library=${library} \
  --dep_add_tags.compiler=${compiler} \
  --record --record_repo=local \
  --record_uoa=mlperf.${division}.${task}.${system}.${model}.${scenario_tag}.${mode_tag}.${dataset_size} \
  --tags=mlperf,${division},${task},${system},${model},${scenario_tag},${mode_tag},${dataset_size} \
  --skip_print_timers --skip_stat_analysis --process_multi_keys
  echo
done # for each model
