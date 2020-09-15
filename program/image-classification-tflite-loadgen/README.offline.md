# MLPerf Inference v0.7 - Image Classification - TFLite

## Offline

- Set up [`program:image-classification-tflite-loadgen`](https://github.com/ctuning/ck-mlperf/blob/master/program/image-classification-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

### ResNet50

#### Performance **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=offline --mode=performance --target_latency=400 \
--verbose --sut=firefly
```

#### Accuracy **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
...
```
