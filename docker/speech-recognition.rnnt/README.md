# MLPerf Inference - Speech Recognition - RNNT

[This collection of images](https://hub.docker.com/r/ctuning/speech-recognition.rnnt) from [dividiti](http://dividiti.com)
tests automated, customizable and reproducible [Collective Knowledge](http://cknowledge.org) workflow for the [Speech Recognition RNNT](https://github.com/mlperf/inference/tree/master/v0.7/speech_recognition/rnnt/) workload. All images include the CK workflow, the latest PyTorch, the PyTorch model, and the (suitably preprocessed) [LibriSpeech](http://www.openslr.org/12/) Dev-Clean dataset.

| `CK_TAG` (`Dockerfile`'s extension)  | Python | GCC   | Comments |
|-|-|-|-|
| `centos-7` | 3.7.8 | 8.3.1 | Updated Python (from 2.7) and GCC (from 4.8) |
| `centos-8` | 3.6.8 | 8.3.1 ||
| `centos-8.python3.7` | 3.7.8 | 8.3.1 | Updated Python (from 3.6) |
| `debian-9`  | 3.5.3 | 6.3.0 | `numba==0.47`, `llvmlite=0.31.0` |
| `debian-10` | 3.7.3 | 8.3.0 ||
| `ubuntu-16.04` | 3.7.8 | 5.4.0 | Updated Python, as 3.5.2 does not support [f-strings](https://www.python.org/dev/peps/pep-0498/). |
| `ubuntu-18.04` | 3.6.9 | 7.5.0 ||
| `ubuntu-20.04` | 3.8.2 | 9.3.0 ||
| `ubuntu-20.04.min` | 3.8.2 | 9.3.0 | Make some steps implicit (comment out with `#-`). |
| `amazonlinux`     | 3.7.6 | 7.3.1 ||
| `amazonlinux.min` | 3.7.6 | 7.3.1 | Make some steps implicit (comment out with `#-`). |
| `amazonlinux.glow` | 3.7.6 | 7.3.1 | Install the Glow compiler and its dependencies. |
| `amazonlinux.glow.min` | 3.7.6 | 7.3.1 | Make some steps implicit (comment out with `#-`). |

It is instructive to diff the following image pairs:
- `centos-7` and `centos-8.python3.7` (as both update Python to the same version).
- `centos-8` and `centos-8.python3.7` (as both are CentOS 8 based).
- `centos-8` and `debian-10` (as they differ mostly in the distro package manager).
- `debian-9` and `debian-10` (as the former differs in some Python 3.5 workarounds).
- `ubuntu-16.04` and `debian-9` (as they are nearly identical).
- `ubuntu-18.04` and `debian-10` (as they are nearly identical).
- `ubuntu-18.04` and `ubuntu-20.04` (as they are nearly identical).
- `ubuntu-20.04` and `ubuntu-20.04.min` (as the latter is derived from the former).
- `amazonlinux` and `centos-8` (as Amazon Linux is similar to CentOS).
- `amazonlinux` and `amazonlinux.min` (as the latter is derived from the former).
- `amazonlinux` and `amazonlinux.glow` (as the latter is derived from the former).
- `amazonlinux.glow` and `amazonlinux.glow.min` (as the latter is derived from the former).

<a name="setup_ck"></a>
## Set up Collective Knowledge

You will need to install [Collective Knowledge](http://cknowledge.org) to build images and save benchmarking results.
Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation) and then pull the ck-mlperf repository:

```bash
$ ck pull repo:ck-mlperf
```

**NB:** Refresh all CK repositories after any updates (e.g. bug fixes):
```bash
$ ck pull all
```


## Build

To build an image e.g. from `Dockerfile.centos-7`:
```bash
$ export CK_IMAGE=speech-recognition.rnnt CK_TAG=centos-7
$ cd `ck find docker:$CK_IMAGE` && docker build -t ctuning/$CK_IMAGE:$CK_TAG -f Dockerfile.$CK_TAG .
```

### Show Python and GCC versions

To show the Python and GCC versions in use in an image built from `Dockerfile.centos-7`:
```bash
$ export CK_IMAGE=speech-recognition.rnnt CK_TAG=centos-7

$ docker run -it --rm ctuning/$CK_IMAGE:$CK_TAG "ck show env --tags=compiler,python"
Env UID:         Target OS: Bits: Name:  Version: Tags:

ef09a59ce5645ffc   linux-64    64 python 3.7.8    64bits,compiler,host-os-linux-64,lang-python,python,target-os-linux-64,v3,v3.7,v3.7.8

$ docker run -it --rm ctuning/$CK_IMAGE:$CK_TAG "ck show env --tags=compiler,gcc"
Env UID:         Target OS: Bits: Name:          Version: Tags:

511106845f6bfe42   linux-64    64 GNU C compiler 8.3.1    64bits,compiler,gcc,host-os-linux-64,lang-c,lang-cpp,target-os-linux-64,v8,v8.3,v8.3.1
```

### Check versions of Python packages

```bash
$ export CK_IMAGE=speech-recognition.rnnt CK_TAG=centos-7
$ docker run -it --rm ctuning/$CK_IMAGE:$CK_TAG \
'ck virtual env --tags=compiler,python \
  --shell_cmd='"'"'$CK_ENV_COMPILER_PYTHON_FILE -m pip show numba'"'"'\
'
Name: numba
Version: 0.48.0
Summary: compiling Python code using LLVM
Home-page: http://numba.github.com
Author: Anaconda, Inc.
Author-email: numba-users@continuum.io
License: BSD
Location: /home/dvdt/.local/lib/python3.7/site-packages
Requires: llvmlite, setuptools, numpy
Required-by: resampy, librosa
```
**NB**: See the quotes magic explained [here](https://stackoverflow.com/questions/1250079/how-to-escape-single-quotes-within-single-quoted-strings).

## Run the default command

To run the default command of an image e.g. built from `Dockerfile.centos-7`:
```bash
$ export CK_IMAGE=speech-recognition.rnnt CK_TAG=centos-7
$ docker run --rm ctuning/$CK_IMAGE:$CK_TAG
```
