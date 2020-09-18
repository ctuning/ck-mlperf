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
--model=ssd-mobilenet --mode=performance --target_latency=1.50 \
--scenario=singlestream --batch_size=1
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --mode=accuracy --dataset_size=5000 \
--scenario=singlestream --batch_size=1
```

#### Compliance **TODO**


<a name="offline"></a>
### Offline

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --mode=performance --target_qps=1250 \
--scenario=offline --batch_size=128
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --mode=accuracy --dataset_size=5000 \
--scenario=offline --batch_size=128
```

#### Compliance **TODO**


<a name="multistream"></a>
### MultiStream

#### Performance ((272,160 == 9! * 3/4) > 270,336)

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --mode=performance --max_query_count=272160 \
--scenario=multistream --batch_size=75 --nstreams={{{batch_size}}}
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=ssd-mobilenet --mode=accuracy --dataset_size=5000 \
--scenario=multistream --batch_size=80 --nstreams={{{batch_size}}}
```

#### Compliance **TODO**
