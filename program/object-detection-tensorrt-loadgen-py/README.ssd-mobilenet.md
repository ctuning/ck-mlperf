# MLPerf Inference v0.7 - Object Detection - TensorRT

- Set up [`program:object-detection-tensorrt-loadgen-py`](https://github.com/ctuning/ck-mlperf/blob/master/program/object-detection-tensorrt-loadgen-py/README.md) on your SUT.
- Customize the examples below for your SUT.

## Prerequisites

```bash
$ ck install package --tags=dataset,object-detection,preprocessed,using-opencv,full,side.300 --ask
```

<a name="ssd-mobilenet"></a>
## SSD-MobileNet

<a name="singlestream"></a>
### SingleStream

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=singlestream \
--mode=performance --target_latency=1.50
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=singlestream \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=singlestream \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_latency=1.50
```


<a name="offline"></a>
### Offline

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=offline --batch_size=128 \
--mode=performance --target_qps=1500
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=offline --batch_size=128 \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=offline --batch_size=128 \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_qps=1500
```


<a name="multistream"></a>
### MultiStream

#### Performance ((272,160 == 9! * 3/4) > 270,336)

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=multistream --batch_size=75 --nstreams={{{batch_size}}} \
--mode=performance --max_query_count=272160
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=multistream --batch_size=75 --nstreams={{{batch_size}}} \
--mode=accuracy --dataset_size=5000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --scenario=multistream --batch_size=75 --nstreams={{{batch_size}}} \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --max_query_count=272160
```
