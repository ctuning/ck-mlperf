#!/bin/bash

division="open"
task="image-classification"
imagenet_size=50000

scenario="singlestream"
scenario_tag="SingleStream"

hostname=`hostname`
if [ "${hostname}" = "hikey961" ]; then
  system="hikey960"
else
  system="${hostname}"
fi

if [ "${system}" = "hikey960" ] || [ "${system}" = "firefly" ]; then
  compiler_tags="gcc,v7"
else
  compiler_tags="gcc,v8"
fi

# Library.
library="tflite-v1.15"
library_tags="tflite,v1.15"

# Image classification models (in the open division).
models=()
models_tags=()
models_preprocessing_tags=()

# Iterate for each model, i.e. resolution and multiplier.
# MobileNet-v1.
#version=1
#resolutions=( 224 192 160 128 )
#multipliers=( 1.0 0.75 0.5 0.25 )
#for resolution in ${resolutions[@]}; do
#  for multiplier in ${multipliers[@]}; do
#    models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
#    models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
#    models_preprocessing_tags+=( "side.${resolution},preprocessed,using-opencv" )
#  done
#done

# MobileNet-v2.
version=2
resolutions=( 224 192 160 128 96 )
multipliers=( 1.0 0.75 0.5 0.25 )
for resolution in ${resolutions[@]}; do
  for multiplier in ${multipliers[@]}; do
    models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
    models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
    models_preprocessing_tags+=( "side.${resolution},preprocessed,using-opencv" )
  done
done
resolutions=( 224 )
multipliers=( 1.4 1.3 )
for resolution in ${resolutions[@]}; do
  for multiplier in ${multipliers[@]}; do
    models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
    models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
    models_preprocessing_tags+=( "side.${resolution},preprocessed,using-opencv" )
  done
done

echo "models=( ${models[@]} )"
echo "models_tags=( ${models_tags[@]} )"
echo "models_preprocessing_tags=( ${models_preprocessing_tags[@]} )"

# Modes.
modes=( "performance" "accuracy" )
modes_tags=( "PerformanceOnly" "AccuracyOnly" )

# Iterate for each model.
for i in $(seq 1 ${#models[@]}); do
  # Configure the model.
  model=${models[${i}-1]}
  model_tags=${models_tags[${i}-1]}
  model_preprocessing_tags=${models_preprocessing_tags[${i}-1]}
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
    # Opportunity to skip by mode.
    #if [ "${mode}" != "accuracy" ]; then continue; fi
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
    ck benchmark program:image-classification-tflite-loadgen \
    --speed --repetitions=1 \
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
