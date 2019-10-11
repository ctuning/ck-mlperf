#!/bin/bash

###############################################################################
# 0. Configure Docker image (build if necessary).
###############################################################################
IMAGE='mlperf-inference-vision-with-ck.tensorrt.ubuntu-18.04'
ck build docker:${IMAGE}


###############################################################################
# 1. Configure paths.
###############################################################################
# TODO: Assume $CK_REPOS is already defined and warn if not.
CK_REPOS=$HOME/CK_REPOS
# Create if if does not exist.
EXPERIMENTS_DIR=/data/$USER/mlperf-inference-vision-experiments
echo "Creating '${EXPERIMENTS_DIR}' for storing experimental results ..."
mkdir -p ${EXPERIMENTS_DIR}


###############################################################################
# 2. Configure scenarios (affects batch options).
###############################################################################
# Uncomment for full run.
scenarios=( 'SingleStream' 'Offline' )
scenarios_lowercase=( 'singlestream' 'offline' )

# Uncomment for debug run.
#scenarios=( 'SingleStream' )
#scenarios_lowercase=( 'singlestream' )

scenarios_selection=()
for scenario in "${scenarios[@]}"; do
  scenarios_selection+=( "--env.CK_SCENARIO=${scenario}" )
done

# FIXME: Invalid batch sizes and counts.
# NB: batch_sizes (#samples/query) must be 1 for SingleStream.
# NB: batch_count (#queries) must be at least 1024 for SingleStream.
# batch_sizes=( 1 2 4 8 16 32 )
batch_sizes=( 1 1 )
batch_count=2


###############################################################################
# 3. Configure models.
# NB: Somewhat counterintuitively, 'models' are actually tags for selecting
# models, while 'models_tags' are tags for recording experimental results.
###############################################################################
# Uncomment for full run.
models=( 'rcnn,nas,lowproposals,vcoco' 'rcnn,resnet50,lowproposals' 'rcnn,resnet101,lowproposals' 'rcnn,inception-resnet-v2,lowproposals' 'rcnn,inception-v2' 'ssd,inception-v2' 'ssd,mobilenet-v1,quantized,mlperf,tf' 'ssd,mobilenet-v1,mlperf,non-quantized,tf' 'ssd,mobilenet-v1,fpn' 'ssd,resnet50,fpn' 'ssdlite,mobilenet-v2,vcoco' 'yolo-v3' )
models_tags=( 'rcnn-nas-lowproposals'  'rcnn-resnet50-lowproposals' 'rcnn-resnet101-lowproposals' 'rcnn-inception-resnet-v2-lowproposals' 'rcnn-inception-v2' 'ssd-inception-v2' 'ssd-mobilenet-v1-quantized-mlperf'    'ssd-mobilenet-v1-non-quantized-mlperf'    'ssd-mobilenet-v1-fpn' 'ssd-resnet50-fpn' 'ssdlite-mobilenet-v2'       'yolo-v3' )

# Uncomment for debug run.
#models=( 'yolo-v3' )
#models_tags=( 'yolo-v3' )

models_selection=()
for model in "${models[@]}"; do
  if [ ${model} = 'yolo-v3' ]
  then
    is_custom_model=1
  else
    is_custom_model=0
  fi
  models_selection+=( "--dep_add_tags.weights=${model} --env.CK_CUSTOM_MODEL=${is_custom_model}" )
done


###############################################################################
# 4. Configure TensorFlow backends.
###############################################################################
# Uncomment for full run.
backends_selection=( '--dep_add_tags.lib-tensorflow=vcpu' '--dep_add_tags.lib-tensorflow=vcuda --env.CUDA_VISIBLE_DEVICES=-1' '--dep_add_tags.lib-tensorflow=vcuda --env.CK_TF_GPU_MEMORY_PERCENT=99' '--dep_add_tags.lib-tensorflow=vcuda --env.CK_TF_GPU_MEMORY_PERCENT=99 --env.CK_ENABLE_TENSORRT=1' '--dep_add_tags.lib-tensorflow=vcuda --env.CK_TF_GPU_MEMORY_PERCENT=99 --env.CK_ENABLE_TENSORRT=1 --env.CK_TENSORRT_DYNAMIC=1' )
backends_tags=( 'cpu-prebuilt' 'cpu' 'cuda' 'tensorrt' 'tensorrt-dynamic' )

# Uncomment for debug run.
#backends_selection=( '--dep_add_tags.lib-tensorflow=vcuda --env.CK_TF_GPU_MEMORY_PERCENT=99 --env.CK_ENABLE_TENSORRT=1 --env.CK_TENSORRT_DYNAMIC=1' )
#backends_tags=( 'tensorrt-dynamic' )


###############################################################################
# 5. Full design space exploration.
###############################################################################
scenarios_len=${#scenarios[@]}
batch_sizes_len=${#batch_sizes[@]}
backends_len=${#backends_selection[@]}
models_len=${#models[@]}

echo "====================="
echo "Starting full DSE ..."
echo "====================="
experiment_idx=1
for i in $(seq 1 ${scenarios_len}); do
  scenario=${scenarios[$i-1]}
  scenario_lowercase=${scenarios_lowercase[$i-1]}
  scenario_selection=${scenarios_selection[$i-1]}
  batch_size=${batch_sizes[$i-1]}
  if [ ${batch_size} = 1 ]
  then
    enable_batch=0
  else
    enable_batch=1
  fi
  batch_selection="--env.CK_ENABLE_BATCH=${enable_batch} --env.CK_BATCH_SIZE=${batch_size} --env.CK_BATCH_COUNT=${batch_count}"
  for j in $(seq 1 ${backends_len}); do
    backend_selection=${backends_selection[$j-1]}
    backend_tags=${backends_tags[$j-1]}
    for k in $(seq 1 ${models_len}); do
      # Model.
      model=${models[$k-1]}
      model_selection=${models_selection[$k-1]}
      model_tags=${models_tags[$k-1]}
      # Profile.
      if [ "${backend_tags}" = 'tensorrt' ] || [ "${backend_tags}" = 'tensorrt-dynamic' ]
      then
        if [ "${model}" = 'yolo-v3' ]
        then
          profile='tf_yolo_trt' # FIXME: tf_trt_yolo?
        else
          profile='default_tf_trt_object_det_zoo' # FIXME: tf_trt_zoo?
        fi
      else
        if [ "${model}" = 'yolo-v3' ]
        then
          profile='tf_yolo'
        else
          profile='default_tf_object_det_zoo' # FIXME: tf_zoo?
        fi
      fi
      profile_selection="--env.CK_PROFILE=${profile}"
      # Record.
      record_uoa="mlperf.object-detection.${scenario_lowercase}.${backend_tags}.${model_tags}"
      record_tags="mlperf,object-detection,${scenario_lowercase},${backend_tags},${model_tags}"
      if [ ${enable_batch} = 1 ]; then
        record_uoa+=".batch-size${batch_size}"
        record_tags+=",batch-size${batch_size}"
      fi
      # Print all parameters.
      echo "experiment_idx: ${experiment_idx}"
      echo "  scenario: ${scenario}"
      echo "  scenario_lowercase: ${scenario_lowercase}"
      echo "  scenario_selection: ${scenario_selection}"
      echo "  batch_size: ${batch_size}"
      echo "  batch_selection: ${batch_selection}"
      echo "  backend_tags: ${backend_tags}"
      echo "  backend_selection: ${backend_selection}"
      echo "  model_tags: ${model_tags}"
      echo "  model: ${model}"
      echo "  model_selection: ${model_selection}"
      echo "  profile: ${profile}"
      echo "  profile_selection: ${profile_selection}"
      echo "  record_uoa=${record_uoa}"
      echo "  record_tags=${record_tags}"
      # NB: Prepend the next line with 'echo' to print the full command without executing it.
      docker run --runtime=nvidia --user=$(id -u):1500 \
      --env-file ${CK_REPOS}/ck-object-detection/docker/${IMAGE}/env.list \
      --volume ${EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
      --rm ctuning/${IMAGE} \
        "ck benchmark program:mlperf-inference-vision --repetitions=1 --env.CK_METRIC_TYPE=COCO \
        ${scenario_selection} ${batch_selection} ${model_selection} ${backend_selection} ${profile_selection} \
        --record --record_repo=local --record_uoa=${record_uoa} --tags=${record_tags}"
      if [ "${?}" != "0" ] ; then
        echo "Error: Failed executing experiment ${experiment_idx} ..."
        exit 1
      fi
      echo "------------------"
      ((experiment_idx++))
    done # for each backend
  done # for each model
done # for each batch size
