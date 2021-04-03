# TensorFlow Lite package

This package installs TensorFlow Lite from the official tagged release `1.15.0-rc2` of TensorFlow ("Heavy").

**NB:** This package is deprecated by `1.15.0`, which includes additional optimizations. It should only
be used to reproduce [MLPerf Inference v0.5 results](https://github.com/mlperf/inference_results_v0.5) from [dividiti](http://dividiti.com).

```bash
$ ck install package:lib-tflite-1.15.0-rc2-src-static --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

## Unresolved issues

### Cannot build for Android (hence removed patches)

Tried installing with `--target_os=android23-arm64` using Android NDK 17.2:
- GCC 4.9.x complained about paths to `stdint.h`.
- LLVM 6.0.2 failed when linking:
```
/home/anton/CK_TOOLS/lib-tflite-src-static-1.15.0-rc2-llvm-android-ndk-6.0.2-android23-arm64/src/tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a(c_api_internal.o): error adding symbols: File in wrong format
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
```

## Resolved issues

# Notes

gfursin fixed problem with eigen library thanks to [this note](https://github.com/tensorflow/tensorflow/issues/43348)
