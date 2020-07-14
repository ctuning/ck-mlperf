#!/bin/bash

#-----------------------------------------------------------------------------#
# MLPerf Inference submission round and submitter.
#-----------------------------------------------------------------------------#
submission="MLPerf Inference v0.7"
echo "Submission round: ${submission}"

submitter="dividiti"
echo "Submitter: ${submitter}"

#-----------------------------------------------------------------------------#
# Image Classification models under the SingleStream scenario on Arm platforms.
#-----------------------------------------------------------------------------#
task="image-classification"
echo "Task: ${task}"

scenario="SingleStream"
scenario_tag="singlestream"
echo "Scenario: ${scenario_tag}"

#-----------------------------------------------------------------------------#
# Configurable parameters.
#-----------------------------------------------------------------------------#
# For the closed division, only run a handful of official MLPerf workloads
# (e.g. ResNet50). For the open division, run the whole gamut.
# Default: closed.
division=${CK_DIVISION:-"closed"}
if [ ${division} != "closed" ]; then
  division="open"
fi
echo "Division: ${division}"

# Dataset size.
# Default: ImageNet validation set size (50,000 images).
imagenet_size=50000
dataset_size=${CK_DATASET_SIZE:-${imagenet_size}}
echo "Dataset size: ${dataset_size}"

# Dataset variation: "full" or e.g. "first.20".
# Default: full.
dataset_variation=${CK_DATASET_VARIATION:-"full"}
echo "Dataset variation: ${dataset_variation}"

# Run workloads under the official MLPerf LoadGen API.
# Default: YES.
use_loadgen=${CK_USE_LOADGEN:-"YES"}
if [ ${use_loadgen} != "YES" ]; then
  use_loadgen="NO"
fi
echo "Use LoadGen: ${use_loadgen}"
loadgen_scenario="--env.CK_LOADGEN_SCENARIO=${scenario}"

# Only print commands but do not execute them.
# Default: NO.
dry_run=${CK_DRY_RUN:-"NO"}
if [ ${dry_run} != "NO" ]; then
  dry_run="YES"
fi
echo "Dry run: ${dry_run}"

# Run only a handful of workloads for testing purposes.
# Default: NO.
quick_run=${CK_QUICK_RUN:-"NO"}
if [ ${quick_run} != "NO" ]; then
  quick_run="YES"
fi
echo "Quick run: ${quick_run}"

# Platform.
hostname=`hostname`
if [ "${hostname}" = "diviniti" ]; then
  # Assume that host "diviniti" is always used to benchmark Android device "mate10pro".
  platform="mate10pro"
  android="--target_os=android24-arm64 --env.CK_LOADGEN_CONF_FILE=user.conf"
elif [ "${hostname}" = "hikey961" ]; then
  platform="hikey960"
  android=""
else
  platform="${hostname}"
  android=""
fi
echo "Platform: ${platform}"

# Platform compiler.
if [ "${platform}" = "mate10pro" ]; then
  # NB: Currently, we only support Clang 6 (NDK 17c) for Android.
  compiler_tags="llvm,v6"
elif [ "${platform}" == "firefly" ] || [ "${platform}" == "xavier" ] || [ "${platform}" == "tx1" ]; then
  compiler_tags="gcc,v7"
elif [ "${platform}" == "rpi4" ]; then
  compiler_tags="gcc,v8"
elif [ "${platform}" == "rpi4coral" ]; then
  compiler_tags="gcc,v9"
else
  compiler_tags="gcc"
fi
echo "Platform compiler tags: ${compiler_tags}"

# Inference engines.
inference_engines=( "tflite" "armnn" )
echo "Inference engines: ( ${inference_engines[@]} )"

# ArmNN backends.
armnn_backend_ref="ref"
armnn_backend_neon="neon"
armnn_backend_opencl="opencl"
select_armnn_backends() {
  if [ "${platform}" == "rpi4" ] || [ "${platform}" == "tx1" ] || [ "${platform}" == "xavier" ]; then
    armnn_backends=( "${armnn_backend_neon}" )
  elif [ "${platform}" == "firefly" ]; then
    armnn_backends=( "${armnn_backend_neon}" "${armnn_backend_opencl}" )
  else
    armnn_backends=( "${armnn_backend_ref}" )
  fi
}
select_armnn_backends
echo "ArmNN backends: ( ${armnn_backends[@]} )"

# Models.
models=()
models_tags=()
models_preprocessing_tags=()
select_models() {
  if [ "${division}" == "closed" ]; then
    models=( "resnet"  )
    models_tags=( "model,tflite,resnet,no-argmax" )
    models_preprocessing_tags=( "${dataset_variation},side.224,preprocessed,using-opencv" )
  elif [ "${division}" == "open" ]; then
    # MobileNet-v1.
    version=1
    if [ "${quick_run}" != "YES" ]; then
      resolutions=( 224 192 160 128 )
      multipliers=( 1.0 0.75 0.5 0.25 )
    else
      resolutions=( 224 )
      multipliers=( 1.0 )
    fi
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        # non-quantized
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
        # quantized
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}-quantized" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},quantized" )
        models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
      done # multiplier
    done # resolution
    # MobileNet-v2.
    version=2
    if [ "${quick_run}" != "YES" ]; then
      resolutions=( 224 192 160 128 96 )
      multipliers=( 1.0 0.75 0.5 0.35 )
    else
      resolutions=( 224 )
      multipliers=( 1.0 )
    fi
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        # non-quantized
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
        # quantized
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}-quantized" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},quantized" )
        models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
      done # multiplier
    done # resolution
    if [ "${quick_run}" != "YES" ]; then
      resolutions=( 224 )
      multipliers=( 1.4 1.3 )
    else
      resolutions=( )
      multipliers=( )
    fi
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        # non-quantized
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
        # quantized
        # NB: mobilenet-v2-1.3-224-quantized is not available: https://github.com/tensorflow/models/issues/8861
        if [ "${multiplier}" != "1.3" ]; then
            models+=( "mobilenet-v${version}-${multiplier}-${resolution}-quantized" )
            models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},quantized" )
            models_preprocessing_tags+=( "${dataset_variation},crop.875,side.${resolution},preprocessed,using-opencv" )
        fi
      done # multiplier
    done # resolution
  else
    echo "ERROR: Unsupported division '${division}'!"
    exit 1
  fi # select models for open or closed
  echo "Models: ( ${models[@]} )"
}
select_models

# Mode: performance, accuracy, submission (?).
if [ ${use_loadgen} == "YES" ]; then
  # Default: performance. TODO: Allow both modes.
  mode=${CK_MODE:-"performance"}
  if [ ${mode} != "performance" ]; then
    mode="accuracy"
  fi
  if [ "${mode}" == "performance" ]; then
    modes=( "performance" )
    loadgen_modes=( "--env.CK_LOADGEN_MODE=PerformanceOnly" )
  elif [ "${mode}" == "accuracy" ]; then
    modes=( "accuracy" )
    loadgen_modes=( "--env.CK_LOADGEN_MODE=AccuracyOnly" )
  else # e.g. "submission"
    modes=( "performance" "accuracy" )
    loadgen_modes=( "--env.CK_LOADGEN_MODE=PerformanceOnly" "--env.CK_LOADGEN_MODE=AccuracyOnly" )
  fi
else
  modes=( "accuracy" )
  loadgen_modes=( "" )
fi
echo "Modes: ( ${modes} )"

echo

experiment_id=1
# Iterate over inference engines.
for inference_engine in ${inference_engines[@]}; do
  if [ "${inference_engine}" == "tflite" ]; then
    inference_engine_version="v2.2.0" # TODO: Iterate over versions which may have different backends.
    inference_engine_program="${task}-tflite"
    inference_engine_backends=( "ruy" )
  elif [ "${inference_engine}" == "armnn" ]; then
    inference_engine_version="rel.20.05"
    inference_engine_program="${task}-armnn-tflite"
    inference_engine_backends=${armnn_backends[@]}
  else
    echo "ERROR: Unsupported inference engine '${inference_engine}'!"
    exit 1
  fi
  if [ "${use_loadgen}" == "YES" ]; then
    loadgen_config_file="--dep_add_tags.loadgen-config-file=${inference_engine_program}"
    inference_engine_program+="-loadgen"
  fi

  # Iterate over inference engine backends.
  for inference_engine_backend in ${inference_engine_backends[@]}; do
    if [ "${inference_engine}" == "armnn" ]; then
      if [ "${inference_engine_backend}" == "${armnn_backend_neon}" ]; then
        armnn_backend="--env.USE_NEON=1"
      elif [ "${inference_engine_backend}" == "${armnn_backend_opencl}" ]; then
        armnn_backend="--env.USE_OPENCL=1"
      elif [ "${inference_engine_backend}" == "${armnn_backend_ref}" ]; then
        armnn_backend=""
      else
        echo "ERROR: Unsupported ArmNN backend '${inference_engine_backend}'!"
        exit 1
      fi # check all armnn backends
    fi

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
        loadgen_mode=${loadgen_modes[${j}-1]}

        if [ "${use_loadgen}" == "YES" ]; then
          batch_count=""
          if [ "${mode}" == "accuracy" ]; then
            dataset_size=${dataset_size}
            buffer_size=500
            verbose=2
          else
            dataset_size=1024
            buffer_size=1024
            verbose=1
          fi
          loadgen_dataset_size="--env.CK_LOADGEN_DATASET_SIZE=${dataset_size}"
          loadgen_buffer_size="--env.CK_LOADGEN_BUFFER_SIZE=${buffer_size}"
        else
          batch_count="--env.CK_BATCH_COUNT=${dataset_size}"
        fi

        # Configure record settings.
        if [ "${use_loadgen}" == "YES" ]; then
          record_uoa="mlperf"
          record_tags="mlperf"
          # Division.
          record_uoa+=".${division}"
          record_tags+=",division.${division}"
          # Submitter.
          record_uoa+=".${submitter}"
          record_tags+=",submitter.${submitter}"
          # Task.
          record_uoa+=".${task}"
          record_tags+=",task.${task}"
        else
          # Task.
          record_uoa="${task}"
          record_tags="task.${task}"
        fi
        # Platform.
        record_uoa+=".${platform}"
        record_tags+=",platform.${platform}"
        # Inference engine.
        record_uoa+=".${inference_engine}-${inference_engine_version}-${inference_engine_backend}"
        record_tags+=",inference_engine.${inference_engine},inference_engine_version.${inference_engine_version},inference_engine_backend.${inference_engine_backend}"
        # Workload.
        record_uoa+=".${model}"
        record_tags+=",workload.${model}"
        # Scenario.
        record_uoa+=".${scenario_tag}"
        record_tags+=",scenario.${scenario_tag}"
        # Mode.
        record_uoa+=".${mode}"
        record_tags+=",mode.${mode}"
        if [ "${mode}" = "accuracy" ]; then
          # Get substring after "preprocessed," to end.
          preprocessing="${model_preprocessing_tags##*preprocessed,}"
          record_uoa+=".${preprocessing}"
          record_tags+=",preprocessing.${preprocessing}"
        fi
        if [ "${mode}" = "accuracy" ] && [ "${dataset_size}" != "${imagenet_size}" ]; then
          record_uoa+=".${dataset_size}"
          record_tags+=",dataset_size.${dataset_size}"
        fi

        # Print experiment description.
        echo "[`date`] Experiment #${experiment_id}"
        experiment_id=$((${experiment_id}+1))
        echo "Inference engine: ${inference_engine}"
        echo "Inference engine version: ${inference_engine_version}"
        echo "Inference engine backend: ${inference_engine_backend}"
        echo "Inference engine program: ${inference_engine_program}"
        echo "Model: ${model}"
        echo "Mode: ${mode}"

        # Skip automatically if experiment record already exists.
        record_dir=$(ck list local:experiment:${record_uoa})
        if [ "${record_dir}" != "" ]; then
          echo "- skipping ..."
          echo
          continue
        fi

        # Skip manuallly e.g. by mode or model.
        #if [ "${mode}" != "performance" ] || [ "${model}" != "mobilenet" ]; then continue; fi

        # Run (but before that print the exact command we are about to run).
        read -d '' CMD <<END_OF_CMD
        ck benchmark program:${inference_engine_program} \
        --speed --repetitions=1 ${armnn_backend} ${batch_count} \
        --env.CK_VERBOSE=${verbose} \
        --dep_add_tags.weights=${model_tags} \
        --dep_add_tags.library=lib,${inference_engine},${inference_engine_version} \
        --dep_add_tags.compiler=${compiler_tags} \
        --dep_add_tags.images=${model_preprocessing_tags} \
        --dep_add_tags.python=v3 \
        --dep_add_tags.mlperf-inference-src=upstream.master \
        ${loadgen_mode} \
        ${loadgen_scenario} \
        ${loadgen_config_file} \
        ${loadgen_dataset_size} \
        ${loadgen_buffer_size} \
        --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
        --skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
        echo "Command:"
        echo ${CMD}
        if [ ${dry_run} == "NO" ]; then
          eval ${CMD}
        fi
        echo
        # Check for errors.
        if [ "${?}" != "0" ]; then
          echo "ERROR: Failed running '${model}' in '${mode}' mode!"
          exit 1
        fi
      done # for each mode
    done # for each model
  done # for each inference engine backend
done # for each inference engine
