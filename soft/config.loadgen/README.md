
# Detecting pre-installed variations of this soft

## Using built-in payloads:

```bash
$	ck detect soft --tags=config,loadgen,image-classification-armnn-tflite

$	ck detect soft --tags=config,loadgen,image-classification-tflite
```

## Using a MLPerf inference git checkout as a dependency:

```bash
$	ck detect soft --tags=config,loadgen,test01

$	ck detect soft --tags=config,loadgen,test04a

$	ck detect soft --tags=config,loadgen,test04b

$	ck detect soft --tags=config,loadgen,test05

$	ck detect soft --tags=config,loadgen,original.mlperf.conf
```

## You can also specify which branch of the MLPerf inference git checkout to detect from:

```bash
$	ck detect soft --tags=config,loadgen,test04b,from.inference.master

$	ck detect soft --tags=config,loadgen,test04a,from.inference.pr518
```