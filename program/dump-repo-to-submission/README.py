# Dump CK repository with LoadGen experiments to MLPerf Inference submission

This Python script originated from automatically converting the Jupyter Notebook that dividiti used
for their MLPerf Inference v0.5 submissions:
```bash
$ jupyter nbconvert --to script dividiti.ipynb
```

It can be compared with the original script as follows:
```bash
$ diff ../../jnotebook/mlperf-inference-v0.5/dividiti.py ./run.py
```

## Usage

```bash
$ ck run ck-mlperf:program:dump-repo-to-submission \
--env.CK_MLPERF_SUBMISSION_REPO=... \
--env.CK_MLPERF_SUBMISSION_ROOT=... \
--env.CK_MLPERF_SUBMISSION_SUBMITTER=...
```
where:
- `CK_MLPERF_SUBMISSION_REPO` is the name of CK repository with experimental results (e.g. `mlperf.closed.image-classification.rpi4.tflite-v1.15`);
- `CK_MLPERF_SUBMISSION_ROOT` is the path to the submission repository (e.g. cloned from https://github.com/mlperf/inference_results_v0.5);
- `CK_MLPERF_SUBMISSION_SUBMITTER` is the submitter string (`dividiti`, by default).

For example:
```bash
$ ck run ck-mlperf:program:dump-repo-to-submission \
--env.CK_MLPERF_SUBMISSION_REPO=mlperf.closed.image-classification.rpi4.tflite-v1.15 \
--env.CK_MLPERF_SUBMISSION_ROOT=$HOME/projects/mlperf/inference_results_v0.5_plus \
--env.CK_MLPERF_SUBMISSION_SUBMITTER=dividiti
```
