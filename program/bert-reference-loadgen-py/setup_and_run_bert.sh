#!/bin/sh

## Usually program's tmp/
#
CWD=`pwd`

export BERT_REF_ROOT=${CK_ENV_MLPERF_INFERENCE}/language/bert
export BERT_BUILD=${CWD}/build
export BERT_BUILD_DATA=${BERT_BUILD}/data

## CWD takes priority over the original code, because we patch some of the original files and retain them in CWD:
#
export PYTHONPATH=${PYTHONPATH}:${CWD}:${BERT_REF_ROOT}:${BERT_REF_ROOT}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT


if [ ! -e "$BERT_BUILD" ]; then
    echo "Build directory does not exist yet"

    mkdir $BERT_BUILD

    if [ -n "$CK_BERT_DATA" ]; then
        echo "Given the following path to be used as build directory: $CK_BERT_DATA"
        if [ ! -e "$CK_BERT_DATA" ]; then
            echo "It did not exist, creating it..."
            mkdir $CK_BERT_DATA
        fi
        ln -s $CK_BERT_DATA $BERT_BUILD/data
    else
        echo "Not given a path for build directory, creating $BERT_BUILD ... Beware, it may take a lot of space!"
        mkdir $BERT_BUILD/data
    fi
    ln -s $CWD ${BERT_BUILD}/logs
    ln -s $CWD ${BERT_BUILD}/result
fi

cp ${BERT_REF_ROOT}/bert_config.json $CWD

OWN_USER_CONF=${CWD}/../user.conf
if [ -e "$OWN_USER_CONF" ]; then
    PATH_TO_USER_CONF=$OWN_USER_CONF
else
    PATH_TO_USER_CONF=${BERT_REF_ROOT}/user.conf
fi

git -C $BERT_REF_ROOT submodule update --init DeepLearningExamples
make -f ${BERT_REF_ROOT}/Makefile download_data
make -f ${BERT_REF_ROOT}/Makefile download_model


## A link for our convenience (not used by the scripts) :
#
rm -f ${CWD}/bert_code
ln -s ${BERT_REF_ROOT} ${CWD}/bert_code


## Adding one line to a Python script without disrupting the indentation structure:
#
OWN_PYTORCH_SUT=${CWD}/pytorch_SUT.py
if [ ! -e "$OWN_PYTORCH_SUT" ]; then
    EXTRA_LINE="if hasattr(self.model.bert, 'set_weights_split'): self.model.bert.set_weights_split()"
    sed "s/^\([\ \t]*\)\(self.model.load_state_dict.*\)/\1\2\n\1${EXTRA_LINE}/ ; s/.cuda()/.to('cpu' if os.getenv('CK_DISABLE_CUDA','') else 'cuda:0')/g" ${BERT_REF_ROOT}/pytorch_SUT.py >$OWN_PYTORCH_SUT
fi

if [ "$CK_LOADGEN_MODE" = "AccuracyOnly" ] || [ "$CK_LOADGEN_MODE" = "Accuracy" ] || [ "$CK_LOADGEN_MODE" = "accuracy" ]; then
    CK_LOADGEN_MODE_STRING="--accuracy"
fi

if [ -n "$CK_LOADGEN_MAX_EXAMPLES" ]; then
    CK_LOADGEN_MAX_EXAMPLES_STRING="--max_examples $CK_LOADGEN_MAX_EXAMPLES"
fi

## Run BERT inference:
#
$CK_ENV_COMPILER_PYTHON_FILE ${BERT_REF_ROOT}/run.py --mlperf_conf=${CK_ENV_MLPERF_INFERENCE}/mlperf.conf --user_conf=${PATH_TO_USER_CONF} --backend=${CK_BERT_BACKEND} --scenario=${CK_LOADGEN_SCENARIO} $CK_LOADGEN_MODE_STRING $CK_LOADGEN_MAX_EXAMPLES_STRING
