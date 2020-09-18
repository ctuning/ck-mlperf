# MLPerf Inference - Object Detection - TFLite (with Coral EdgeTPU support)

### CPU (SSD-MobileNet-v1)

#### Performance

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,ssd-mobilenet --variation_prefix=from-zenodo --separator=:` \
--scenario=singlestream --mode=performance --target_latency=123 \
--verbose --sut=rpi4coral
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --library=tflite-v2.2.0-ruy \
--model:=`ck list_variations misc --query_module_uoa=package --tags=model,tflite,ssd-mobilenet --variation_prefix=from-zenodo --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=50 \
--verbose --sut=rpi4coral
```

### Coral EdgeTPU (SSD-MobileNet-v2)

#### Performance

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --library=tflite-edgetpu \
--model:=`ck list_variations misc --query_module_uoa=package --tags=ssd-mobilenet,edgetpu --variation_prefix=v2 --separator=:` \
--scenario=singlestream --mode=performance --target_latency=123 \
--verbose --sut=rpi4coral
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --library=tflite-edgetpu \
--model:=`ck list_variations misc --query_module_uoa=package --tags=ssd-mobilenet,edgetpu --variation_prefix=v2 --separator=:` \
--scenario=singlestream --mode=accuracy --dataset_size=5000 \
--verbose --sut=rpi4coral
```
