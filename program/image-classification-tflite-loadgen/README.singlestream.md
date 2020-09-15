# MLPerf Inference v0.7 - Image Classification - TFLite

## SingleStream

- Set up [`program:image-classification-tflite-loadgen`](https://github.com/ctuning/ck-mlperf/blob/master/program/image-classification-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

### ResNet50

#### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=88 \
--verbose --sut=xavier
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=xavier
...
accuracy=76.442%, good=38221, total=50000
```

#### Compliance **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=88 \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 \
--verbose --sut=xavier
```

### MobileNet-v1

#### Performance **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=xavier
```

#### Accuracy **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=xavier
```
#### Compliance **TODO**


### MobileNet-v2

#### Performance **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=xavier
```

#### Accuracy **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=xavier
```

#### Compliance **TODO**


### MobileNet-v3

#### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=performance --target_latency=6 \
--verbose --sut=xavier
```

#### Accuracy **NOT TESTED**

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=xavier
```

#### Compliance **TODO**


### EfficientNet

#### Performance **NOT TESTED**

```bash
$ ck gen cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=10 \
--verbose --sut=xavier
```

#### Accuracy **NOT TESTED**

```bash
$ ck gen cmdgen:benchmark.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=xavier
```

#### Compliance **TODO**
