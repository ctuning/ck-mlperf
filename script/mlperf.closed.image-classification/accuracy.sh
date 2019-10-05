#!/bin/bash

division='closed'
task='image-classification'
imagenet_size=50000

scenario='SingleStream'
scenario_tag='singlestream'

hostname=`hostname`
if [ ${hostname} = 'hikey961' ]
then
  system='hikey960'
else
  system=${hostname}
fi

if [ ${system} = 'hikey960' ] || [ ${system} = 'firefly' ]
then
  compiler='gcc,v7'
else
  compiler='gcc,v8'
fi

mode='AccuracyOnly'
mode_tag='accuracy'
if [ ${mode_tag} = 'accuracy' ]
then
  dataset_size=50000
  buffer_size=500
else
  dataset_size=1024
  buffer_size=1024
fi

# Library.
library='tflite,v1.15'
library_tag='tflite-v1.15'

# Image classification models (in the closed division).
models=( 'mobilenet' 'resnet' )
models_tags=( 'mobilenet,non-quantized' 'resnet' )
# Preferred preprocessing methods per model.
preprocessing_methods=( 'preprocessed,using-opencv' 'preprocessed,using-tensorflow' )

# Iterate for each model.
for i in $(seq 1 ${#models[@]}); do
  # Configure for the model.
  model=${models[${i}-1]}
  model_tags=${models_tags[${i}-1]}
  # Use the same preprocessing method for all models.
  preprocessing_method=${preprocessing_methods[0]}
  # Configure record settings.
  record_uoa="mlperf.${division}.${task}.${system}.${library_tag}.${model}.${scenario_tag}.${mode_tag}"
  record_tags="mlperf,${division},${task},${system},${library_tag},${model},${scenario_tag},${mode_tag}"
  if [ ${mode} = 'accuracy' ] && [ "${dataset_size}" != "${imagenet_size}" ]; then
    record_uoa+=".${dataset_size}"
    record_tags+=",${dataset_size}"
  fi
  # Benchmark.
  echo "Benchmarking '${model}' ..."
  echo ck benchmark program:image-classification-tflite-loadgen \
  --speed --repetitions=1 --env.CK_VERBOSE=2 \
  --env.CK_LOADGEN_SCENARIO=${scenario} \
  --env.CK_LOADGEN_MODE=${mode} \
  --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
  --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
  --dep_add_tags.weights=${model_tags} \
  --dep_add_tags.library=${library} \
  --dep_add_tags.compiler=${compiler} \
  --dep_add_tags.images=${preprocessing_method} \
  --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
  --skip_print_timers --skip_stat_analysis --process_multi_keys
  echo
done # for each model
