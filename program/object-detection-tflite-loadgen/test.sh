#! /bin/bash


export PATH=/home/gavin/CK/ck-env/platform.init/rpi4:$PATH


. /home/gavin/CK/local/env/8fcd7e4840447f88/env.sh
. /home/gavin/CK/local/env/2b56d24f11ad01b6/env.sh
. /home/gavin/CK/local/env/d4d350501403a89b/env.sh
. /home/gavin/CK/local/env/5cd643808e0eda85/env.sh
. /home/gavin/CK/local/env/0bb11cf577d14b9c/env.sh
. /home/gavin/CK/local/env/4a61146e09bd9705/env.sh
. /home/gavin/CK/local/env/8f072ba14506a007/env.sh
. /home/gavin/CK/local/env/40ec4bee8de6cb1b/env.sh
. /home/gavin/CK/local/env/9e9602ec7d048318/env.sh
. /home/gavin/CK/local/env/787a024416376177/env.sh
. /home/gavin/CK/local/env/c60b46b6ba158021/env.sh
. /home/gavin/CK/local/env/9748e996a4285401/env.sh

. /home/gavin/CK/local/env/8fcd7e4840447f88/env.sh 1

export CK_ANNOTATIONS_OUT_DIR=annotations
export CK_BATCH_COUNT=20
export CK_BATCH_SIZE=1
export CK_DETECTIONS_OUT_DIR=detections
export CK_PREPROCESSED_OUT_DIR=preprocessed
export CK_RESULTS_OUT_DIR=results
export CK_SILENT_MODE=0
export CK_SKIP_IMAGES=0
export CK_TIMER_FILE=tmp-ck-timer.json
export USE_NEON=NO
export USE_OPENCL=NO


echo    executing code ...
./detect
