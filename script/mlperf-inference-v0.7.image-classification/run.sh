#!/bin/bash

task="image-classification"
imagenet_size=50000

# Configurable variables for various scenarios
# dry_run mode, default to disabled
dry_run=${CK_DRY_RUN:-""}
echo "Dry run: ${dry_run}"

# "closed" or "open" division, default to "closed"
division=${CK_DIVISION:-"closed"}
echo "Division: ${division}"

# Loadgen or no loadgen, default to loadgen 
use_loadgen=${CK_LOADGEN:-""}
echo "Use Loadgen: ${use_loadgen}"

# quick run for testing. Disabled by default
quick_run=${CK_QUICK_RUN:-""}
echo "Quick Run: ${quick_run}"

# Mode. Defaults to performance
run_mode=${CK_MODE:-"performance"}
echo "Mode: $run_mode}"

scenario="singlestream"
scenario_tag="SingleStream"

library_tflite_android="tflite-v1.13"
library_tflite_android_tags="tflite,v1.13"
library_tflite_linux="tflite-v1.15.3"
library_tflite_linux_tags="tflite,v1.15.3"
library_armnn="armnn-v20.05"
library_armnn_tags="armnn,tflite,neon,opencl,rel.20.05"

# Implementation.
# TODO: Add iteration over implementations and backends. (Now, simply define which one is active.)
implementation_tflite="image-classification-tflite"
implementation_tflite_loadgen="image-classification-tflite-loadgen"
implementation_armnn="image-classification-armnn-tflite"
implementation_armnn_loadgen="image-classification-armnn-tflite-loadgen"
implementation_frontends=()

# ArmNN backends.
implementation_armnn_backend_ref="ref"
implementation_armnn_backend_neon="neon"
implementation_armnn_backend_opencl="opencl"
implementation_backends=( ${implementation_armnn_backend_ref} ${implementation_armnn_backend_neon} )

if [ -z "${use_loadgen}" ] ; then
  implementation_frontends=( ${implementation_tflite_loadgen} ${implementation_armnn_loadgen} ) 
else
  implementation_frontends=( ${implementation_tflite} ${implementation_armnn} )
fi



# Global variables for blacklist function
system=""
implementation_backends_usable=()

# Function to remove blacklisted backends
# and update global variable of possible backends
backends_selector(){

  backend_blacklist=() 
  # Add opencl to blacklist option for the targets "rpi4" Raspberry Pi 4
  # "tx1" for Nvidia Tx1. "dummy_target" also mentioned for further usage
  if [ "${system}" == "rpi4" ] || [ ${system} == "tx1" ] || [ $system == "dummy_target" ]; then
    backend_blacklist+=( ${backend_opencl} )
  fi

  # Add neon option to blacklist for dummy_target
  if [ "$system" == "dummy_target" ] ; then
    backend_blacklist+=( ${backend_neon} )
  fi

  implementation_backends_usable=${implementation_backends[@]}
  # implementation backends usable for current system should
  # remove the blacklisted backends from full list of backends

  #echo ${implementation_backends[@]}
  #echo ${backend_blacklist[@]}

  for backend_bl in ${backend_blacklist[@]}; 
  do 
    implementation_backends_usable=( ${implementation_backends_usable[@]/$backend_bl} )
  done

}

# variables for models selection
# depending on division
models=()
models_tags=()
models_preprocessing_tags=()

# select the appropriate list of models from a list
# and update relevant global variables
# call this function after "implementation" has been 
# chosen in the for-loop
division_models_selector(){

  models=()
  models_tags=()
  models_preprocessing_tags=()

  if [ "$division" == "closed" ]; then 
    models=( "resnet"  ) 
    models_tags=( "model,tflite,resnet,no-argmax" )
    models_preprocessing_tags=( "full,side.224,preprocessed,using-opencv" )
  else 
    division="open"
    # MobileNet-v1.
    version=1
  if [ "${quick_run}" == "" ]; then
    resolutions=( 224 192 160 128 )
    multipliers=( 1.0 0.75 0.5 0.25 )
  else 
    resolutions=( 224 )
    multipliers=( 1.0 )
  fi
  for resolution in ${resolutions[@]}; do
    for multiplier in ${multipliers[@]}; do
      models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
      models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
      models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
      if [ "${implementation}" == "${implementation_tflite}" ] || [ "${implementation}" == "${implementation_tflite_loadgen}" ] ; then
        models+=( "mobilenet-v${version}-${multiplier}-${resolution}-quantized" )
        models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},quantized" )
        models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
      fi
    done
  done
  if [ "${quick_run}" == "" ]; then
    resolutions=( 224 192 160 128 96 )
    multipliers=( 1.0 0.75 0.5 0.35 )
  else 
    resolutions=( 224 )
    multipliers=( 1.0 )
  fi
  # MobileNet-v2.
  version=2
  for resolution in ${resolutions[@]}; do
    for multiplier in ${multipliers[@]}; do
      models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
      models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
      models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
    done
  done
  if [ "${quick_run}" == "" ]; then
    resolutions=( 224 )
    multipliers=( 1.4 1.3 )
  else 
    resolutions=( 224 )
    multipliers=( 1.4 )
  fi
  for resolution in ${resolutions[@]}; do
    for multiplier in ${multipliers[@]}; do
      models+=( "mobilenet-v${version}-${multiplier}-${resolution}" )
      models_tags+=( "model,tflite,mobilenet,v${version}-${multiplier}-${resolution},non-quantized" )
      models_preprocessing_tags+=( "full,crop.875,side.${resolution},preprocessed,using-opencv" ) # "first.20,crop.875,side.${resolution},preprocessed,using-opencv"
    done
  done
 fi

}


# ------------ Main script starts here ---------------




# System.
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


# call backend selection function
backends_selector


# Compiler.
if [ "${system}" = "mate10pro" ]; then
  # NB: Currently, we only support Clang 6 (NDK 17c) for Android.
  compiler_tags="llvm,v6"
elif [ "${system}" = "hikey960" ] || [ "${system}" = "firefly" ]; then
  compiler_tags="gcc,v7"
else
  compiler_tags="gcc,v8"
fi



# Modes defaults to performance otherwise select accuracy
if [ "${run_mode}" == "performance" ]; then
  modes=( "performance" )
  modes_tags=( "PerformanceOnly" )
else
  modes=( "accuracy" )
  modes_tags=( "AccuracyOnly" )
fi




 # Iterate over implementation frontends
for implementation in ${implementation_frontends[@]}; do
echo "  Implementation frontend is ${implementation}"

  # call model selection function at this point since
  # mobilenets tested differ on implementation i.e. tflite 
  # has extra mobilenets to test 

  division_models_selector


  # Iterate over implementation backends
  for implementation_armnn_backend in ${implementation_backends_usable[@]}; do
    echo "  Implementation backend is ${implementation_armnn_backend}"

   # Library.
    if [ "${implementation}" == "${implementation_tflite}" ] || [ "${implementation}" == "${implementation_tflite_loadgen}" ]; then
      if [ "${android}" != "" ]; then
        # Android supports different tflite version. The latest doesn't work
        library=${library_tflite_android}
        library_tags=${library_tflite_android_tags}
      else
        library=${library_tflite_linux}
        library_tags=${library_tflite_linux_tags}
      fi
      armnn_backend=""
    elif [ "${implementation}" == "${implementation_armnn}" ] || [ "${implementation}" == "${implementation_armnn_loadgen}" ]; then
      library=${library_armnn}
      library_tags=${library_armnn_tags}
      if [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_opencl}" ]; then
        armnn_backend="--env.USE_OPENCL=1"
      elif [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_neon}" ]; then
        armnn_backend="--env.USE_NEON=1"
      elif [ "${implementation_armnn_backend}" == "${implementation_armnn_backend_ref}" ]; then
        armnn_backend=""
      else
        echo "ERROR: Unsupported ArmNN backend '${implementation_armnn_backend}'!"
        exit 1
      fi
    else
      echo "ERROR: Unsupported implementation '${implementation}'!"
      exit 1
    fi

   

    # Iterate for each model.
    for i in $(seq 1 ${#models[@]}); do
      # Configure the model.
      model=${models[${i}-1]}
      model_tags=${models_tags[${i}-1]}
      echo "  Implementing ${model}"
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
        echo "Current mode is ${mode}"
        if [ "${mode}" = "accuracy" ]; then
          dataset_size=50000
          buffer_size=500
          verbose=2
        else
          dataset_size=1024
          buffer_size=1024
          verbose=1
        fi
        if [ "${implementation}" == "${implementation_armnn}" ]; then
          batch_count="--env.CK_BATCH_COUNT=${dataset_size}"
        else
          batch_count=""
        fi
        # Opportunity to skip by mode or model.
        #if [ "${mode}" != "performance" ] || [ "${model}" != "mobilenet" ]; then continue; fi
        # Configure record settings.
        record_uoa="mlperf.${division}.${task}.${system}.${library}"
        record_tags="mlperf,${division},${task},${system},${library}"
        if [ "${implementation}" == "${implementation_armnn_loadgen}" ]; then
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
        # Run (but before that print the exact command we are about to run).
        echo "Running '${model}' in '${mode}' mode ..."
        read -d '' CMD <<END_OF_CMD
        ck benchmark program:${implementation} \
        --speed --repetitions=1 ${android} ${armnn_backend} ${batch_count} \
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
        --dep_add_tags.mlperf-inference-src=upstream.master \
        --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
        --skip_print_timers --skip_stat_analysis --process_multi_keys
END_OF_CMD
        echo ${CMD}
        if [ -z ${dry_run} ]; then 
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
  done
done



