#!/bin/bash

division="closed"
task="image-classification"
imagenet_size=50000

scenario="singlestream"
scenario_tag="SingleStream"

hostname=`hostname`
if [ "${hostname}" = "diviniti" ]; then
  # Assume that host "diviniti" is always used to benchmark Android device "mate10pro".
  system="mate10pro"
  android="--target_os=android24-arm64 --env.CK_LOADGEN_CONF_FILE=user.conf"
elif [ "${hostname}" = "hikey961" ]; then
  system="hikey960"
  android=""
else
  system="${hostname}"
  android=""
fi

if [ "${system}" = "hikey960" ] || [ "${system}" = "firefly" ]; then
  compiler_tags="gcc,v7"
else
  compiler_tags="gcc,v8"
fi

# Library.
library="tflite-v1.15"
library_tags="tflite,v1.15"

# Implementation.
implementation="image-classification-tflite-loadgen"

# Image classification models (in the closed division).
models=( "mobilenet" "resnet" )
models_tags=( "model,tflite,mobilenet-v1-1.0-224,non-quantized" "model,tflite,resnet,no-argmax" )
# Preferred preprocessing methods per model.
models_preprocessing_tags=( "full,side.224,preprocessed,using-opencv" "full,side.224,preprocessed,using-tensorflow" )

# Modes.
modes=( "performance" "accuracy" )
modes_tags=( "PerformanceOnly" "AccuracyOnly" )

# Iterate for each model.
for i in $(seq 1 ${#models[@]}); do
  # Configure the model.
  model=${models[${i}-1]}
  model_tags=${models_tags[${i}-1]}
  # Configure the preprocessing method.
  if [ "${system}" = "hikey960" ]; then
    model_preprocessing_tags=${models_preprocessing_tags[${i}-1]}
  else
    # By default, use the same preprocessing method for all models.
    model_preprocessing_tags=${models_preprocessing_tags[0]}
  fi
  # Iterate for each mode.
  for j in $(seq 1 ${#modes[@]}); do
    # Configure the mode.
    mode=${modes[${j}-1]}
    mode_tag=${modes_tags[${j}-1]}
    if [ "${mode}" = "accuracy" ]; then
      dataset_size=50000
      buffer_size=500
      verbose=2
    else
      dataset_size=1024
      buffer_size=1024
      verbose=1
    fi
    # Opportunity to skip by mode or model.
    #if [ "${mode}" != "performance" ] || [ "${model}" != "mobilenet" ]; then continue; fi
    # Configure record settings.
    record_uoa="mlperf.${division}.${task}.${system}.${library}.${model}.${scenario}.${mode}"
    record_tags="mlperf,${division},${task},${system},${library},${model},${scenario},${mode}"
    if [ "${mode}" = "accuracy" ]; then
      # Get substring after "preprocessed," to end.
      preprocessing="${model_preprocessing_tags##*preprocessed,}"
      record_uoa+=".${preprocessing}"
      record_tags+=",${preprocessing}"
    fi
    if [ "${mode}" = "accuracy" ] && [ "${dataset_size}" != "${imagenet_size}" ]; then
      record_uoa+=".${dataset_size}"
      record_tags+=",${dataset_size}"
    fi
    # Run (but before that print the exact command we are about to run).
    echo "Running '${model}' in '${mode}' mode ..."
    read -d '' CMD <<END_OF_CMD
    ck benchmark program:${implementation} \
    --speed --repetitions=1 ${android} \
    --env.CK_VERBOSE=${verbose} \
    --env.CK_LOADGEN_SCENARIO=${scenario_tag} \
    --env.CK_LOADGEN_MODE=${mode_tag} \
    --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
    --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
    --dep_add_tags.weights=${model_tags} \
    --dep_add_tags.library=${library_tags} \
    --dep_add_tags.compiler=${compiler_tags} \
    --dep_add_tags.images=${model_preprocessing_tags} \
    --dep_add_tags.python=v3 \
    --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
    --skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
    echo ${CMD}
    eval ${CMD}
    echo
    # Check for errors.
    if [ "${?}" != "0" ]; then
      echo "Error: Failed running '${model}' in '${mode}' mode ..."
      exit 1
    fi
  done # for each mode
done # for each model
