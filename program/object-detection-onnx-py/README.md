# TensorFlow object-detection program

## Pre-requisites

### Repositories

```bash
$ ck pull repo:ck-mlperf
```

### ONNX libraries

```bash
$ ck install package --tags=lib,onnx
$ ck install package --tags=lib,onnxruntime
```

### ONNX Object Detection model
Install one or more object detection model package:
```bash
$ ck install package --tags=model,object-detection,onnx
```

### Datasets
```bash
$ ck install package --tags=dataset,object-detection,preprocessed,side.1200
```

## Running

```bash
$ ck run object-detection-onnx-py
```

### Program parameters

#### `CK_BATCH_COUNT`

The number of images to be processed.

Default: `1`

#### `CK_BATCH_COUNT`

The number of skiped images.

Default: `0`
