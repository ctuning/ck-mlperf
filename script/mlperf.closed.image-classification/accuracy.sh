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
models_tags=( 'mobilenet,non-quantized' 'resnet' )
# Preferred preprocessing methods per model.
preprocessing_methods=( 'preprocessed,using-opencv' 'preprocessed,using-tensorflow' )
# Iterate for each model.
for i in $(seq 1 ${#models[@]}); do
  model=${models[${i}-1]}
  model_tags=${models_tags[${i}-1]}
  preprocessing_method=${preprocessing_methods[0]} # use the same method for all models
  echo "Benchmarking '${model}' ..."
  ck benchmark program:image-classification-tflite-loadgen \
  --speed --repetitions=1 \
  --env.CK_LOADGEN_SCENARIO=${scenario} \
  --env.CK_LOADGEN_MODE=${mode} \
  --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
  --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
  --env.CK_VERBOSE=2 \
  --dep_add_tags.weights=${model_tags} \
  --dep_add_tags.library=${library} \
  --dep_add_tags.compiler=${compiler} \
  --dep_add_tags.images=${preprocessing_method} \
  --record --record_repo=local \
  --record_uoa=mlperf.${division}.${task}.${system}.${model}.${scenario_tag}.${mode_tag}.${dataset_size} \
  --tags=mlperf,${division},${task},${system},${model},${scenario_tag},${mode_tag},${dataset_size} \
  --skip_print_timers --skip_stat_analysis --process_multi_keys
  echo
done # for each model
