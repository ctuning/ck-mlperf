#!/bin/sh

export BERT_REF_ROOT=${CK_ENV_MLPERF_INFERENCE}/language/bert
export BERT_BUILD=./build
export BERT_BUILD_DATA=${BERT_BUILD}/data
export PYTHONPATH=${PYTHONPATH}:${BERT_REF_ROOT}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT


if [ ! -e "$BERT_BUILD" ]; then
    echo "Build directory does not exist yet"

    if [ -n "$CK_BERT_REF_BUILD_DIR" ]; then
        echo "Given the following path to be used as build directory: $CK_BERT_REF_BUILD_DIR"
        if [ ! -e "$CK_BERT_REF_BUILD_DIR" ]; then
            echo "It did not exist, creating it..."
            mkdir $CK_BERT_REF_BUILD_DIR
        fi
        ln -s $CK_BERT_REF_BUILD_DIR $BERT_BUILD
    else
        echo "Not given a path for build directory, creating $BERT_BUILD ... Beware, it may take a lot of space!"
        mkdir $BERT_BUILD
    fi
fi

CWD=`pwd`

cp ${BERT_REF_ROOT}/bert_config.json $CWD
cp ${BERT_REF_ROOT}/user.conf $CWD

make -f ${BERT_REF_ROOT}/Makefile download_data
make -f ${BERT_REF_ROOT}/Makefile download_model

cp ${BERT_REF_ROOT}/create_squad_data.py ${BERT_REF_ROOT}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT/utils/create_squad_data.py

sed -i'' 's/ accuracy-squad.py/ .\/accuracy-squad.py/g' ${BERT_REF_ROOT}/run.py

rm -rf build/logs build/result bert_code

ln -s $CWD build/logs
ln -s $CWD build/result
ln -s ${BERT_REF_ROOT} bert_code

$CK_ENV_COMPILER_PYTHON_FILE ${BERT_REF_ROOT}/run.py --mlperf_conf=${CK_ENV_MLPERF_INFERENCE}/mlperf.conf --user_conf=${BERT_REF_ROOT}/user.conf --backend=${CK_BERT_BACKEND} --scenario=${CK_LOADGEN_SCENARIO} $CK_LOADGEN_MODE_STRING
