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
--model=resnet50 --mode=performance --target_latency=2.0 \
--scenario=singlestream --batch_size=1
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --mode=accuracy --dataset_size=50000 \
--scenario=singlestream --batch_size=1
```

#### Compliance **TODO**


<a name="multistream"></a>
### MultiStream

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --mode=performance --max_query_count=1440 \
--scenario=multistream --batch_size=69 --nstreams={{{batch_size}}}
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --mode=accuracy --dataset_size=50000 \
--scenario=multistream --batch_size=69 --nstreams={{{batch_size}}}
```

#### Compliance **TODO**


<a name="offline"></a>
### Offline

#### Performance

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --mode=performance --target_qps=2000 \
--scenario=offline --batch_size=70
```

#### Accuracy

```bash
$ ck run cmdgen:benchmark.tensorrt-loadgen --verbose \
--model=resnet50 --mode=accuracy --dataset_size=50000 \
--scenario=offline --batch_size=70
```

#### Compliance **TODO**
