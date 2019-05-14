## Ubuntu 18.04

**NB:** `#` means execution under root or with `sudo`.

### Build image
```
# docker build -t ubuntu-18.04 .
```

### Run image

#### Image Classification (default)
```
# docker run --rm ubuntu-18.04
```
**NB:** Equivalent to:
```
# docker run --rm ubuntu-18.04 \
ck run program:image-classification-tflite
```

#### Image Classification (custom)
```
# docker run --rm ubuntu-18.04 \
ck run program:image-classification-tflite --env.CK_BATCH_COUNT=10
```

#### Shell (needs changing Dockerfile)
To allow running shell, in `Dockerfile` change:
```
ENTRYPOINT ["/bin/bash"]
CMD ["ck run program:image-classification-tflite"]
```
to:
```
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["ck run program:image-classification-tflite"]
```
and run as:
```
# docker run --rm ubuntu-18.04 bash
```
