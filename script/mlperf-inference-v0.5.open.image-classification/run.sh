#!/bin/bash

division="open"
task="image-classification"
imagenet_size=50000

# Scenarios.
scenario="singlestream"
scenario_tag="SingleStream"

# Modes.
modes=( "performance" "accuracy" )
modes_tags=( "PerformanceOnly" "AccuracyOnly" )

# Implementations.
implementation_tflite="image-classification-tflite-loadgen"
implementation_armnn="image-classification-armnn-tflite-loadgen"
implementation_armnn_no_loadgen="image-classification-armnn-tflite"
implementations=( "${implementation_armnn}" "${implementation_tflite}" )
# ArmNN backends.
implementation_armnn_backend_ref="ref"
implementation_armnn_backend_neon="neon"
implementation_armnn_backend_opencl="opencl"
implementation_armnn_backends=( "${implementation_armnn_backend_opencl}" "${implementation_armnn_backend_neon}" )
# Dummy ArmNN backend for TFLite.
implementation_armnn_backend_dummy="dummy"

# System.
hostname=`hostname`
if [ "${hostname}" = "diviniti" ]; then
  # Assume that host "diviniti" is always used to benchmark Android device "mate10pro".
  system="mate10pro"
  android="--target_os=android24-arm64"
elif [ "${hostname}" = "hikey961" ]; then
  system="hikey960"
  android=""
else
  system="${hostname}"
  android=""
fi

# Compiler.
if [ "${system}" = "mate10pro" ]; then
  # NB: Currently, we only support Clang 6 (NDK 17c) for Android.
  compiler_tags="llvm,v6"
elif [ "${system}" = "hikey960" ] || [ "${system}" = "firefly" ]; then
  compiler_tags="gcc,v7"
else
  compiler_tags="gcc,v8"
fi

experiment_id=1
# Iterate for each implementation.
for implementation in ${implementations[@]}; do
  # Select library and backends based on implementation.
  if [ "${implementation}" == "${implementation_tflite}" ]; then
    config_tag="image-classification-tflite"
    if [ "${android}" != "" ]; then
      # NB: Currently, we only support TFLite v1.13 for Android.
      library="tflite-v1.13"
      library_tags="tflite,v1.13"
    else
      library="tflite-v1.15"
      library_tags="tflite,v1.15"
    fi
    implementation_armnn_backends=( "${implementation_armnn_backend_dummy}" )
  elif [ "${implementation}" == "${implementation_armnn}" ] || [ "${implementation}" == "${implementation_armnn_no_loadgen}" ]; then
    config_tag="image-classification-armnn-tflite"
    library="armnn-v19.08"
    library_tags="armnn,tflite,neon,opencl,rel.19.08"
    if [ "${system}" = "rpi4" ]; then
      # NB: Only use Neon backend on Raspberry Pi 4.
      implementation_armnn_backends=( "${implementation_armnn_backend_neon}" )
      library_tags="armnn,tflite,neon,rel.19.08"
    fi
  else
    echo "ERROR: Unsupported implementation '${implementation}'!"
    exit 1
  fi
  # Iterate for each backend.
  for implementation_armnn_backend in ${implementation_armnn_backends[@]}; do
    if [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_ref}" ] || [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_dummy}" ]; then
      armnn_backend=""
    elif [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_neon}" ]; then
      armnn_backend="--env.USE_NEON=1"
    elif [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_opencl}" ]; then
      armnn_backend="--env.USE_OPENCL=1"
    else
      echo "ERROR: Unsupported ArmNN backend '${implementation_armnn_backend}'!"
      exit 1
    fi
    # Create a list of MobileNets-v1/v2 models depending on the implementation.
    models=()
    models_tags=()
    models_preprocessing_tags=()
    # MobileNet-v1.
    version=1
    resolutions=( 224 192 160 128 )
    multipliers=( 1.0 0.75 0.5 0.25 )
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
        if [ "${implementation}" == "${implementation_tflite}" ]; then
          models+=( "mobilenet-v${version}-${multiplier}-${resolution}-quantized" )
          models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},quantized" )
          models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
        fi
      done
    done
    # MobileNet-v2.
    version=2
    resolutions=( 224 192 160 128 96 )
    multipliers=( 1.0 0.75 0.5 0.35 )
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
      done
    done
    resolutions=( 224 )
    multipliers=( 1.4 1.3 )
    for resolution in ${resolutions[@]}; do
      for multiplier in ${multipliers[@]}; do
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
        models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
      done
    done
    # Iterate for each model.
    for i in $(seq 1 ${#models[@]}); do
      # Configure the model.
      model=${models[${i}-1]}
      model_tags=${models_tags[${i}-1]}
      model_preprocessing_tags=${models_preprocessing_tags[${i}-1]}
      echo "model_preprocessing_tags=${model_preprocessing_tags}"

      # Iterate for each mode.
      for j in $(seq 1 ${#modes[@]}); do
        # Configure the mode.
        mode=${modes[${j}-1]}
        mode_tag=${modes_tags[${j}-1]}
        if [ "${mode}" == "accuracy" ]; then
          dataset_size=50000
          buffer_size=500
          verbose=2
        else
          dataset_size=1024
          buffer_size=1024
          verbose=1
        fi

        # Configure record settings.
        record_uoa="mlperf.${division}.${task}.${system}.${library}"
        record_tags="mlperf,${division},${task},${system},${library}"
        if [ "${implementation}" == "${implementation_armnn}" ]; then
          record_uoa+=".${implementation_armnn_backend}"
          record_tags+=",${implementation_armnn_backend}"
        fi
        record_uoa+=".${model}.${scenario}.${mode}"
        record_tags+=",${model},${scenario},${mode}"
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

        echo "[`date`] Experiment #"${experiment_id}": ${record_uoa} ..."
        experiment_id=$((${experiment_id}+1))

        # Skip automatically if experiment record already exists.
        record_dir=$(ck list local:experiment:${record_uoa})
        if [ "${record_dir}" != "" ]; then
          echo "[`date`] - skipping ..."
          echo
          continue
        fi

        # Skip manually.
        if [ "${implementation}" == "${implementation_armnn}" ] || [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_neon}" ] ; then
          echo "[`date`] - skipping ..."
          echo
          continue
        fi

        # Run (but before that print the exact command we are about to run).
        echo "[`date`] - running ..."
        read -d '' CMD <<END_OF_CMD
        ck benchmark program:${implementation} \
        --speed --repetitions=1 ${android} ${armnn_backend} \
        --env.CK_VERBOSE=${verbose} \
        --env.CK_LOADGEN_SCENARIO=${scenario_tag} \
        --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} \
        --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
        --env.CK_LOADGEN_MODE=${mode_tag} \
        --dep_add_tags.loadgen-config-file=${config_tag} \
        --dep_add_tags.weights=${model_tags} \
        --dep_add_tags.library=${library_tags} \
        --dep_add_tags.compiler=${compiler_tags} \
        --dep_add_tags.images=${model_preprocessing_tags} \
        --dep_add_tags.python=v3 \
        --dep_add_tags.mlperf-inference-src=upstream.master \
        --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
        --skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
        echo ${CMD}
        eval ${CMD}
        # Check for errors.
        if [ "${?}" != "0" ]; then
          echo "ERROR: Failed running '${model}' with '${implementation}'!"
          exit 1
        fi
        echo
      done # for each mode
    done # for each model
  done # for each implementation backend
done # for each implementation
# - 130 performance/accuracy pairs of experiments on firefly/hikey960: (54 models with tflite + 38 models with armnn-neon + 38 models with armnn-opencl).
# - 92 performance/accuracy pairs of experiments on rpi4: (54 models with tflite + 38 models with armnn-neon).
echo "[`date`] Done."
