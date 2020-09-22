# MLPerf Inference v0.7 - Image Classification - ArmNN

## SingleStream

- Set up [`program:image-classification-armnn-tflite-loadgen`](https://github.com/ctuning/ck-mlperf/blob/master/program/image-classification-armnn-tflite-loadgen/README.md) on your SUT.
- Customize the examples below for your SUT.

### ResNet50

#### Performance

##### Neon

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=400 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-opencl \
--model=resnet50 --scenario=singlestream --mode=performance --target_latency=400 \
--verbose --sut=firefly
```

#### Accuracy

##### Neon

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
...
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-opencl \
--model=resnet50 --scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
...
```

#### Compliance

##### Neon

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model=resnet50 --scenario=singlestream  --compliance,=TEST04-A,TEST04-B,TEST05,TEST01 \
--verbose --sut=firefly
```

##### OpenCL

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-opencl \
--model=resnet50 --scenario=singlestream --compliance,=TEST04-A,TEST04-B,TEST05,TEST01 \
--verbose --sut=firefly
```


### MobileNet-v1

#### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v1 --variation_prefix=v1- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

### MobileNet-v2

#### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=5 \
--verbose --sut=firefly
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v2 --variation_prefix=v2- --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

### MobileNet-v3

#### Performance

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=performance --target_latency=6 \
--verbose --sut=firefly
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,mobilenet-v3 --variation_prefix=v3- --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```

### EfficientNet

#### Performance

```bash
$ ck gen cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=performance --target_latency=10 \
--verbose --sut=firefly
```

#### Accuracy

```bash
$ ck gen cmdgen:benchmark.tflite-loadgen --library=armnn-v20.08-neon \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,effnet --variation_prefix=lite --separator=:` \
--model_extra_tags,=non-quantized,quantized \
--scenario=singlestream --mode=accuracy --dataset_size=50000 \
--verbose --sut=firefly
```
