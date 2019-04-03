# Common scripts for benchmarking programs

Common preprocessing and postprocessing scripts to be used in benchmarking
programs such as `ck-tensorflow:program:object-detection-tflite`, etc.

A client program has to reference scripts in its meta in the section `run_time`, e.g.:

```json
  "run_cmds": {
    "default": {
      "run_time": {
        "pre_process_via_ck": {
          "data_uoa":       "24c98b0cee248d93",
          "module_uoa":     "script",
          "script_name":    "preprocess"
        },
        "post_process_via_ck": {
          "data_uoa":       "24c98b0cee248d93",
          "module_uoa":     "script",
          "script_name":    "postprocess"
        },

        "run_cmd_main": "$#BIN_FILE#$",
```

It is supposed that the client program provides required enviroment variables and dependencies with suitable names that scripts will search for.


## Preprocessing

The preprocessing script prepares images for a client program.

Preprocessing steps:

- Read the required number of images from a dataset. The number of images is governed by program parameters `CK_BATCH_COUNT` and `CK_BATCH_SIZE`.
  
- Prepare images for loading into a model. Preparation includes scaling images to a size defined by the input images size of the model being benchmarked. See the section **Input preprocessing parameters** below.

- Store prepared images into a cache directory.

As a result, the preprocessing script provides a set of variables (in separate `env.ini` file) that a client program should use:

- `MODEL_DATASET_TYPE` - point to dataset type which the model was trained on (`coco` for example)
- `MODEL_TFLITE_GRAPH` - path to model's `.tflite` file
- `MODEL_IMAGE_CHANNELS` - number of color channels in image. Default value: 3
- `MODEL_IMAGE_HEIGHT`, `MODEL_IMAGE_WIDTH` - the sizes of an image, which it should have to be processed on by the model
- `MODEL_NEED_BACKGROUND_CORRECTION` - indicate that model don't consider "background" class (True or False)
- `MODEL_NORMALIZE_DATA` - indicate that image should be "normalized" before processing (True or False)
- `MODEL_SUBTRACT_MEAN` - indicate that operation "subtract mean" should be executed on the image before processing

- `DETECTIONS_OUT_DIR` - path to results of detections (text files, one per image)
- `PREPROCESS_OUT_DIR` - path to a directory containing preprocessed images.
- `PREPROCESSED_FILES` - path to a file containing list of images to be processed with their original sizes `file.mame;width;height`. This file contains only image file names without paths.

- `IMAGE_COUNT` - count of images which will be taken from dataset to process. Default value: 1.
- `BATCH_SIZE` - size of batch (number of images that can be processed "simultaneously"). Now batch processing by TF Lite is not supporting. Should be: 1

- `NUMBER_OF_PROCESSORS` - indicate how many threads TF Lite can use to process images. Default value: 1.
- `FULL_REPORT` - additional logging level (True or False)
- `VERBOSE` - maximal logging level (True or False)


### Program-specific preprocessing

A client program may contain its own additional preprocessing script  `preprocess-next.py`. If it exists, it will be called after the common preprocessing.

## Images dataset

A client program should provide access to the dataset via run-time dependency `dataset`, e.g.:

```json
  "run_deps": {
    "dataset": {
      "local": "yes",
      "name": "Object detection dataset",
      "sort": 20,
      "tags": "dataset,object-detection"
    },
```

## Weights package

The model for benchmarking is provided by a client program's run-time dependency `weights`, e.g.:

```json
  "run_deps": {
    "weights": {
      "force_target_as_host": "yes",
      "local": "yes",
      "name": "TensorFlow model",
      "sort": 60,
      "tags": "ssd,tflite,model,object-detection"
    }
```


## Program parameters

Here we describe a client program's parameters affecting the pre/post-processig stages.

### Input image parameters



#### `CK_BATCH_COUNT`

The number of processed images

Default: `1`.

#### `CK_SKIP_IMAGES`

The number of images which will be skiped before processed images

Default: `0`.
