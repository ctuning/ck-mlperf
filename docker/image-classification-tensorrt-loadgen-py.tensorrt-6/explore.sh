#!/bin/bash

echo "Tunable parameters:"

# Batch sizes.
batch_sizes=( ${CK_BATCH_SIZES:-$(seq 1 2)} ) # use parentheses to interpret the string as an array
echo "- batch sizes: ${batch_sizes[@]}"

echo

echo "Configuration parameters:"

# Docker image name.
image=${CK_IMAGE:-"image-classification-tensorrt-loadgen-py.tensorrt-6"}
echo "- image: '${image}'"

# Path to CK repos.
repos=${CK_REPOS:-"$HOME/CK"}
echo "- CK repos: '${repos}'"

# Path to experiment entries.
experiments_dir=${CK_EXPERIMENTS_DIR:-$HOME/tensorrt-experiments}
echo "- experiments dir: '${experiments_dir}'"
mkdir -p ${experiments_dir}

# Dry run - print commands but do not execute them.
dry_run=${CK_DRY_RUN:-""}
echo "- dry run: ${dry_run}"

# Skip existing experiments.
skip_existing=${CK_SKIP_EXISTING:-""}
echo "- skip existing: ${skip_existing}"

# Task: image-classification or object-detection.
task=${CK_TASK:-"image-classification"}
echo "- task: ${task}"

# LoadGen program.
program="${task}-tensorrt-loadgen-py"
program_dir="/home/dvdt/CK_REPOS/ck-mlperf/program/${program}"
echo "- program: '${program}'"
echo "- program directory: '${program_dir}'"

# Platform: velociti, xavier, ...
platform=${CK_PLATFORM:-"velociti"}
echo "- platform: ${platform}"

# Model name for LoadGen config: resnet50, ssd-resnet34, mobilenet, ssd-mobilenet, gnmt.
model_name=${CK_LOADGEN_MODEL_NAME:-"resnet"}
echo "- model name: ${model_name}"

# Model tags.
model_tags=${CK_MODEL_TAGS:-"resnet,converted-from-onnx,maxbatch.32"}
echo "- model tags: ${model_tags}"

# LoadGen scenario: SingleStream, MultiStream, Server, Offline.
scenario=${CK_LOADGEN_SCENARIO:-MultiStream}
if [ "${scenario}" = "SingleStream" ]; then
  scenario_tag="singlestream"
elif [ "${scenario}" = "MultiStream" ]; then
  scenario_tag="multistream"
elif [ "${scenario}" = "Server" ]; then
  scenario_tag="server"
elif [ "${scenario}" = "Offline" ]; then
  scenario_tag="offline"
else
  echo "ERROR: Unsupported LoadGen scenario '${scenario}'!"
  exit 1
fi
echo "- scenario: ${scenario} (${scenario_tag})"

# LoadGen mode: PerformanceOnly, AccuracyOnly.
mode=${CK_LOADGEN_MODE:-PerformanceOnly}
if [ "${mode}" = "PerformanceOnly" ]; then
  mode_tag="performance"
elif [ "${mode}" = "AccuracyOnly" ]; then
  mode_tag="accuracy"
else
  echo "ERROR: Unsupported LoadGen mode '${mode}'!"
  exit 1
fi
echo "- mode: ${mode} (${mode_tag})"

if [ "${task}" = "image-classification" ]; then
  imagenet_size=500 # 50000
  if [ "${mode}" = "AccuracyOnly" ]; then
    dataset_size=${CK_LOADGEN_DATASET_SIZE:-${imagenet_size}}
    buffer_size=${CK_LOADGEN_BUFFER_SIZE:-500}
  else
    dataset_size=${CK_LOADGEN_DATASET_SIZE:-${imagenet_size}}
    buffer_size=${CK_LOADGEN_BUFFER_SIZE:-1024}
  fi
elif [ "${task}" = "object-detection" ]; then
  coco_size=5000
  if [ "${mode}" = "AccuracyOnly" ]; then
    dataset_size=${CK_LOADGEN_DATASET_SIZE:-${coco_size}}
    buffer_size=${CK_LOADGEN_BUFFER_SIZE:-50}
  else
    if [ "${model_name}" = "ssd-mobilenet" ]; then
      dataset_size=${CK_LOADGEN_DATASET_SIZE:-256}
      buffer_size=${CK_LOADGEN_BUFFER_SIZE:-256}
    elif [ "${model_name}" = "ssd-resnet34" ]; then
      dataset_size=${CK_LOADGEN_DATASET_SIZE:-64}
      buffer_size=${CK_LOADGEN_BUFFER_SIZE:-64}
    else
      dataset_size=${CK_LOADGEN_DATASET_SIZE:-1024}
      buffer_size=${CK_LOADGEN_BUFFER_SIZE:-1024}
    fi # model name
  fi # mode
else
  echo "ERROR: Unsupported task '${task}'!"
  exit 1
fi # task
echo "- dataset size: ${dataset_size}"
echo "- buffer size: ${buffer_size}"

# In the PerformanceOnly mode, affects the number of samples per query that LoadGen issues
# (aiming to meet the minimum duration of 60 seconds and, in the Offline mode, the minimum
# number of samples of 24,576).
target_qps=${CK_LOADGEN_TARGET_QPS:-70}
if [ "${mode}" = "PerformanceOnly" ]; then
  if [ "${scenario}" == "SingleStream" ] || [ "${scenario}" == "Offline" ]; then
    TARGET_QPS="--env.CK_LOADGEN_TARGET_QPS=${target_qps}"
  fi
fi
echo "- target QPS (queries per second): ${target_qps} ('${TARGET_QPS}')"

# Allow to override the number of queries in the PerformanceOnly mode.
# By default, use 1440=720*2:
# - 720==6! (6 factorial) is evenly divisible between any number of workers 1-6.
# - 1200==60*20 is the minimum number of 50 ms queries to meet the minimum duration of 60 ms.
# - 1440 > 1200.
count_override=${CK_LOADGEN_COUNT_OVERRIDE:-1440}
if [ "${mode}" = "PerformanceOnly" ]; then
  COUNT_OVERRIDE="--env.CK_LOADGEN_COUNT_OVERRIDE=${count_override}"
fi
echo "- count override: ${count_override} ('${COUNT_OVERRIDE}')"

# Numerical precision of the TensorRT model.
precision=${CK_PRECISION:-fp32}
echo "- precision: ${precision}"

# Input preprocessing.
preprocessing_tags=${CK_PREPROCESSING_TAGS:-"rgb8,side.224,preprocessed,using-opencv"}
echo "- preprocessing tags: ${preprocessing_tags}"

# Timestamp.
timestamp=$(date +%Y%m%d-%H%M%S)
echo "- timestamp: ${timestamp}"

# File to print after each iteration.
file_to_print=${CK_FILE_TO_PRINT:-""}
if [ -z "${file_to_print}" ]; then
  if [ "${mode_tag}" = "accuracy" ]; then
    file_to_print="accuracy.txt"
  else
    file_to_print="mlperf_log_summary.txt"
  fi
fi

# Prepare record UOA and tags.
mlperf="mlperf"
division="closed"
library="tensorrt"
benchmark=${model_name}
record_uoa="${mlperf}.${division}.${task}.${platform}.${library}.${benchmark}.${scenario_tag}.${mode_tag}"
record_tags="${mlperf},${division},${task},${platform},${library},${benchmark},${scenario_tag},${mode_tag}"
if [ "${mode_tag}" = "accuracy_" ]; then # FIXME: This part is intentionally disabled for the time being.
  # Get substring after "preprocessed," to end, i.e. "using-opencv" here.
  preprocessing="${preprocessing_tags##*preprocessed,}"
  record_uoa+=".${preprocessing}"
  record_tags+=",${preprocessing}"
  if [ "${task}" = "image-classification" ] && [ "${dataset_size}" != "${imagenet_size}" ]; then
    record_uoa+=".${dataset_size}"
    record_tags+=",${dataset_size}"
  fi
  if [ "${task}" = "object-detection" ] && [ "${dataset_size}" != "${coco_size}" ]; then
    record_uoa+=".${dataset_size}"
    record_tags+=",${dataset_size}"
  fi
fi
echo "- record UOA: ${record_uoa}"
echo "- record tags: ${record_tags}"

# Blank line before printing commands.
echo

# Skip existing experiments if requested.
if (ck find experiment:${record_uoa} >/dev/null) && [[ "${skip_existing}" ]]; then
  echo "Experiment '${record_uoa}' already exists, skipping ..."
  exit 0
fi

# Explore tunable parameters.
for batch_size in ${batch_sizes[@]}; do
  read -d '' CMD <<END_OF_CMD
  docker run --runtime=nvidia \
  --env-file ${repos}/ck-mlperf/docker/${image}/env.list \
  --user=$(id -u):1500 \
  --volume ${experiments_dir}:/home/dvdt/CK_REPOS/local/experiment \
  --rm ctuning/${image} \
    "ck benchmark program:${program} --repetitions=1 --env.CK_SILENT_MODE \
    --env.CK_LOADGEN_MODE=${mode} --env.CK_LOADGEN_SCENARIO=${scenario} \
    --env.CK_BATCH_SIZE=${batch_size} --env.CK_LOADGEN_MULTISTREAMNESS=${batch_size} \
    --env.CK_LOADGEN_COUNT_OVERRIDE=${count_override} \
    --env.CK_LOADGEN_DATASET_SIZE=${dataset_size} --env.CK_LOADGEN_BUFFER_SIZE=${buffer_size} \
    --env.CK_LOADGEN_CONF_FILE=${program_dir}/user.conf \
    --dep_add_tags.weights=model,tensorrt,${model_tags},${precision} \
    --dep_add_tags.lib-python-tensorrt=python-package,tensorrt \
    --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags} \
    --skip_print_timers --skip_stat_analysis --process_multi_keys \
  && echo '--------------------------------------------------------------------------------' \
  && echo '${file_to_print}' \
  && echo '--------------------------------------------------------------------------------' \
  && cat  /home/dvdt/CK_REPOS/ck-mlperf/program/${program}/tmp/${file_to_print} \
  && echo ''"
END_OF_CMD
  echo ${CMD}
  if [ -z "${dry_run}" ]; then
    eval ${CMD}
  fi
  echo
done


if [ -z "${dry_run}" ]; then
  echo "Done."
else
  echo "Done (dry run)."
fi
