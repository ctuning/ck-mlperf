# MLPerf Inference v0.7 - Object Detection - TensorRT

- Set up [`program:object-detection-tensorrt-loadgen-py`](https://github.com/ctuning/ck-mlperf/blob/master/program/object-detection-tensorrt-loadgen-py/README.md) on your SUT.
- Customize the examples below for your SUT.

## Prerequisites

```bash
$ ck install package --tags=dataset,object-detection,preprocessed,using-opencv,full,side.1200 --ask
```

<a name="ssd-resnet34"></a>
## SSD-ResNet34

<a name="singlestream"></a>
### SingleStream

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=singlestream \
 --mode=performance --target_latency=29.43
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=singlestream \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=singlestream \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_latency=29.43
```


<a name="offline"></a>
### Offline

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=offline --batch_size=2 \
--mode=performance --target_qps=50
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=offline --batch_size=2 \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=offline --batch_size=2 \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_qps=50
```

<a name="multistream"></a>
### MultiStream

#### Performance ((272,160 == 9! * 3/4) > 270,336)

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=multistream --batch_size=2 --nstreams={{{batch_size}}} \
--mode=performance --max_query_count=272160
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=multistream --batch_size=2 --nstreams={{{batch_size}}} \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-resnet34 --scenario=multistream --batch_size=2 --nstreams={{{batch_size}}} \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --max_query_count=272160
```
