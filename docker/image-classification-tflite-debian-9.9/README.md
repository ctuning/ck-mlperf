## [Debian](https://hub.docker.com/_/debian/) 9.9 ("stretch")

**NB:** `#` means execution under root or with `sudo`.

### Change directory
```
$ cd `ck find docker:image-classification-tflite-debian-9.9`
```

### Build image
```
# docker build -f Dockerfile -t image-classification-tflite-debian-9.9 .
```

### Check image 
#### View layer-by-layer build history
```
# docker history image-classification-tflite-debian-9.9
```
#### View space usage
```
# docker system df -v
```

### Run image

#### Image Classification (default)
```
# docker run --rm image-classification-tflite-debian-9.9
```
**NB:** Equivalent to:
```
# docker run --rm image-classification-tflite-debian-9.9 \
"ck run program:image-classification-tflite"
```

#### Image Classification (custom)
```
# docker run --rm image-classification-tflite-debian-9.9 \
"ck run program:image-classification-tflite --env.CK_BATCH_COUNT=10"
```

#### Bash
```
# docker run -it --rm image-classification-tflite-debian-9.9 bash
```
