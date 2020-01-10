# Dump CK repository with LoadGen experiments to MLPerf Inference submission

This Python script originated from automatically converting the Jupyter Notebook that dividiti used
for generating data for the [interactive MLPerf Inference v0.5 dashboard](http://cknowledge.org/dashboard/mlperf.inference):
```bash
$ jupyter nbconvert --to script results.ipynb
```

It can be compared with the original script as follows:
```bash
$ diff ../../jnotebook/mlperf-inference-v0.5/results.py ./run.py
```

## Usage

```bash
$ ck run ck-mlperf:program:dump-submissions-to-dashboard \
--env.CK_MLPERF_SUBMISSION_ROOT=... \
--env.CK_MLPERF_DASHBOARD_FILE=... \
--env.CK_MLPERF_DASHBOARD_DIR=...
```
where:
- `CK_MLPERF_SUBMISSION_ROOT` is the path to an input repository with submissions (e.g. cloned from https://github.com/mlperf/inference_results_v0.5);
- `CK_MLPERF_DASHBOARD_FILE` is the name of an output dashboard file (`mlperf-inference-v0.5-results.zip`, by default).
- `CK_MLPERF_DASHBOARD_DIR` is the name of an output dashboard directory. If empty (by default), dump into in the dashboard module directory (`ck find ck-mlperf:module:mlperf.inference`).

### Examples

#### Default

Dump `mlperf-inference-v0.5-results.zip` from `$HOME/projects/mlperf/inference_results_v0.5_dividiti` into the dashboard directory:

```bash
$ ck run ck-mlperf:program:dump-submissions-to-dashboard
```


#### Custom 1

Dump `mlperf-inference-unofficial-results.zip` from `$HOME/projects/mlperf/inference_results_v0.5_plus` into the dashboard directory:

```bash
$ ck run ck-mlperf:program:dump-submissions-to-dashboard \
--env.CK_MLPERF_SUBMISSION_ROOT=$HOME/projects/mlperf/inference_results_v0.5_plus \
--env.CK_MLPERF_DASHBOARD_FILE=mlperf-inference-unofficial-results.zip
```


#### Custom 2

Dump `mlperf-inference-unofficial-results.zip` from `$HOME/projects/mlperf/inference_results_v0.5_plus` into the current directory:

```bash
$ ck run ck-mlperf:program:dump-submissions-to-dashboard \
--env.CK_MLPERF_SUBMISSION_ROOT=$HOME/projects/mlperf/inference_results_v0.5_plus \
--env.CK_MLPERF_DASHBOARD_FILE=mlperf-inference-unofficial-results.zip \
--env.CK_MLPERF_DASHBOARD_DIR=`pwd`
```
