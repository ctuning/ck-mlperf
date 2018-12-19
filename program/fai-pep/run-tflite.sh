#!/bin/bash

cd ${CK_ENV_BENCH_FAI_PEP}

${CK_ENV_COMPILER_PYTHON_FILE} benchmarking/run_bench.py \
  --framework "tflite" \
  -b ${CK_ENV_BENCH_FAI_PEP_MODEL} \
  --repo_dir ${CK_ENV_LIB_TF}/src
