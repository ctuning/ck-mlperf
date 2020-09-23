# MLPerf Inference v0.7 - Image Classification - TensorRT

- Set up [`program:image-classification-tensorrt-loadgen-py`](https://github.com/ctuning/ck-mlperf/blob/master/program/image-classification-tensorrt-loadgen-py/README.md) on your SUT.
- Customize the examples below for your SUT.

<a name="resnet50"></a>
## ResNet50

<a name="singlestream"></a>
### SingleStream

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=singlestream \
--mode=performance --target_latency=2.04
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=singlestream \
--mode=accuracy --dataset_size=50000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=singlestream \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_latency=2.04
```


<a name="offline"></a>
### Offline

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=offline --batch_size=64 \
--mode=performance --target_qps=1400
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=offline --batch_size=64 \
--mode=accuracy --dataset_size=50000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=offline --batch_size=64 \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --target_qps=1400
```


<a name="multistream"></a>
### MultiStream

#### Performance ((272,160 == 9! * 3/4) > 270,336)

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=multistream --batch_size=70 --nstreams={{{batch_size}}} \
--mode=performance --max_query_count=272160
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=multistream --batch_size=70 --nstreams={{{batch_size}}} \
--mode=accuracy --dataset_size=50000
```

#### Compliance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --scenario=multistream --batch_size=70 --nstreams={{{batch_size}}} \
--compliance,=TEST04-A,TEST04-B,TEST01,TEST05 --max_query_count=272160
```
