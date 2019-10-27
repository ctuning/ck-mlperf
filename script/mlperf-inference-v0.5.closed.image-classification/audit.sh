#!/bin/bash

division="closed"
task="image-classification"
#imagenet_size=50000
#record_uoa_tail=".V2"

# Scenarios.
scenario="singlestream"
scenario_tag="SingleStream"

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
  android="--target_os=android24-arm64 --env.CK_LOADGEN_CONF_FILE=user.conf" # FIXME: Pick the right config for audit tests.
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

# Image classification models (for the closed division).
models=( "mobilenet" "resnet" )
models_tags=( "model,tflite,mobilenet-v1-1.0-224,non-quantized" "model,tflite,resnet,no-argmax" )
models_target_latency_ms=( 50 200 ) # Expected latencies with ArmNN-OpenCL on HiKey960.

# Audit tests.
v05_dir=$( ck cat env --tags=mlperf,inference,source,upstream.master | grep CK_ENV_MLPERF_INFERENCE_V05 | cut -d'=' -f2 )
audit_dir="${v05_dir}/audit/nvidia"
audit_tests=( "TEST01" "TEST03" "TEST04-A" "TEST04-B" "TEST05" )

# Iterate for each implementation.
for implementation in ${implementations[@]}; do
  # Select library and backends based on implementation.
  if [ "${implementation}" == "${implementation_tflite}" ]; then
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
    # Iterate for each model.
    for i in $(seq 1 ${#models[@]}); do
      # Configure the model.
      model=${models[${i}-1]}
      model_tags=${models_tags[${i}-1]}
      model_target_latency_ms=${models_target_latency_ms[${i}-1]}

      # Iterate for each audit test.
      for audit_test in ${audit_tests[@]}; do
        # Select LoadGen config file.
        audit_config="${audit_dir}"/"${audit_test}"/audit.config
        if test -f "${audit_config}"; then
          config="${audit_config}"
        else
          config="../user.conf" # TEST03
        fi
        # TODO: Document how to install/detect datasets.
        if [ "${audit_test}" = "TEST03" ]; then
          model_preprocessing_tags="full,side.224,preprocessed,using-opencv,audit.test03"
          mode_tag="SubmissionRun"
        else
          model_preprocessing_tags="full,side.224,preprocessed,using-opencv,crop.875"
          mode_tag="PerformanceOnly"
        fi

        # Configure record settings.
        record_uoa="mlperf.${division}.${task}.${system}.${library}"
        record_tags="mlperf,${division},${task},${system},${library}"
        if [ "${implementation}" == "${implementation_armnn}" ]; then
          record_uoa+=".${implementation_armnn_backend}"
          record_tags+=",${implementation_armnn_backend}"
        fi
        record_uoa+=".${model}.${scenario}.audit.${audit_test}${record_uoa_tail}"
        record_tags+=",${model},${scenario},audit,${audit_test}"

        # Opportunity to skip.
        if [ "${audit_test}" != "TEST03" ]; then continue; fi
        if [ "${implementation}" == "${implementation_tflite}" ] &&  [ "${model}" == "resnet" ]; then continue; fi

        # Run (but before that print the exact command we are about to run).
        echo "Running '${model}' for audit '${audit_test}' with '${implementation}' ..."
        read -d '' CMD <<END_OF_CMD
        ck benchmark program:${implementation} \
        --speed --repetitions=1 ${android} ${armnn_backend} \
        --env.CK_LOADGEN_SCENARIO=${scenario_tag} \
        --env.CK_LOADGEN_CONF_FILE=${config} \
        --env.CK_LOADGEN_MODE=${mode_tag} \
        --env.CK_LOADGEN_SINGLE_STREAM_TARGET_LATENCY_MS=${model_target_latency_ms} \
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
        echo
        # Check for errors.
        if [ "${?}" != "0" ]; then
          echo "ERROR: Failed running '${model}' for audit '${audit_test}' with '${implementation}'!"
          exit 1
        fi
      done # for each audit test
    done # for each model
  done # for each implementation backend
done # for each implementation
echo "Done: `date`"
