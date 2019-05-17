## [Ubuntu](https://hub.docker.com/_/ubuntu/) 18.04

**NB:** `#` means execution under root or with `sudo`.

### Change directory
```
$ cd `ck find docker:image-classification-tflite-ubuntu-18.04`
```

### Build image
```
# docker build . -f Dockerfile -t image-classification-tflite-ubuntu-18.04
```

### Check image 
#### View layer-by-layer build history
```
# docker history image-classification-tflite-ubuntu-18.04
```
#### View space usage
```
# docker system df -v
```

### Run image

#### Image Classification (default)
```
# docker run --rm image-classification-tflite-ubuntu-18.04
```
**NB:** Equivalent to:
```
# docker run --rm image-classification-tflite-ubuntu-18.04 \
"ck run program:image-classification-tflite"
```

#### Image Classification (custom)
```
# docker run --rm image-classification-tflite-ubuntu-18.04 \
"ck run program:image-classification-tflite --env.CK_BATCH_COUNT=10"
```

#### Bash
```
# docker run -it --rm image-classification-tflite-ubuntu-18.04 bash
```
