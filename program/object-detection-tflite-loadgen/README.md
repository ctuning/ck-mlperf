# MLPerf Inference - Object Detection - TFLite (with Coral EdgeTPU support)

## Coral EdgeTPU

### SSD-MobileNet-v1-EdgeTPU, SSD-MobileNet-v2-EdgeTPU

#### Performance

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --verbose \
--library=tflite-edgetpu --model:=v1:v2 \
--scenario=singlestream --mode=performance --target_latency=20 \
--sut=rpi4coral
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --verbose \
--library=tflite-edgetpu --model:=v1:v2 \
--scenario=singlestream --mode=accuracy --dataset_size=5000 \
--sut=rpi4coral
```

## CPU

### SSD-MobileNet-v1 non-quantized

#### Performance

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --verbose \
--library=tflite-v2.2.0-ruy --model:=non-quantized \
--scenario=singlestream --mode=performance --target_latency=170 \
--sut=rpi4coral
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.object-detection.tflite-loadgen --verbose \
--library=tflite-v2.2.0-ruy --model:=non-quantized \
--scenario=singlestream --mode=accuracy --dataset_size=5000 \
--sut=rpi4coral
```
