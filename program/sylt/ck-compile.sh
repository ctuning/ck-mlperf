#!/bin/sh

cwd=$(pwd)

cd ${CK_ENV_BENCH_SYLT_TRACE_GENERATOR}

rm -f ./a.out

echo ""
echo "$CK_CXX $CK_COMPILER_FLAGS_OBLIGATORY $CK_FLAGS_DYNAMIC_BIN ${CK_FLAG_PREFIX_INCLUDE}./ demo.cc  ${CK_FLAGS_OUTPUT}a.out"
$CK_CXX $CK_COMPILER_FLAGS_OBLIGATORY $CK_FLAGS_DYNAMIC_BIN ${CK_FLAG_PREFIX_INCLUDE}./ demo.cc ${CK_FLAGS_OUTPUT}a.out
er=$?; if [ $er != 0 ]; then exit $er; fi

echo ""
cp -f ./a.out $cwd
