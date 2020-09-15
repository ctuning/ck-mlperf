# MLPerf Inference v0.7 - Image Classification - TFLite

## ResNet50

### SingleStream

- Set up [`program:image-classification-tflite-loadgen`](https://github.com/ctuning/ck-mlperf/blob/master/program/image-classification-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

##### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=123 \
--sut=xavier
```

##### Accuracy

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--sut=xavier
...
```

##### Compliance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=123 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 --sut=xavier
```

## MobileNet

**TODO**

