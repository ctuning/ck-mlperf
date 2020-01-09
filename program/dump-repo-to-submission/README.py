This Python script originated from automatically converting the Jupyter Notebook that dividiti used
for their MLPerf Inference v0.5 submissions:
```bash
$ jupyter nbconvert --to script dividiti.ipynb
```

It can be compared with the original script as follows:
```bash
$ diff ../../jnotebook/mlperf-inference-v0.5/dividiti.py ./run.py
```
