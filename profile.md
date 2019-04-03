How to trace down pytorch's CUDA performance bottleneck
==

python level
---

#### requirements
1. snakeviz
2. cProfile
3. line-profiler ( optional )
```bash
pip install snakeviz cprofile
```
>Note: if you profile the program with python3's cProfile, you must use the snakeviz of python3. Otherwise snakeviz raises a warning message saying the profiler output is not a cProfile file

Since the CUDA tensor operations in pytorch are all asynchronous. Normally the results returned by naive cprofiler will give you some wrong results. Fortunately we can enforce the cuda calls to be synchronized by simply setting the environment variable `CUDA_LAUNCH_BLOCKING=1`.

```bash
CUDA_LAUNCH_BLOCKING=1 python -m cProfile -o program.prof program.py
```

```bash
# if you runs locally, the command will automaticall yopen a web browser
snakeviz program.prof

# normally we run our program on remote server, addtional parameters are
# required to access from internet
snakeviz -s -H 0.0.0.0 -p <YOUR_PORT>

# you may get detailed informtion from help page
snakeviz --help
```

- ( optional ) `line-profiler` for pure CLI profiling
Instead of `snakeviz`, you can use `line-profiler` to get the time spend on each line of codes. As noted before, you should also set the `CUDA_LAUNCH_BLOCKING=1` for sensible profiling results.

To use `line-profiler`, you should modify a few lines in your code, simply place a `@profile` decorator above the funtions or methods of insterest.
```bash
pip install line-profiler
CUDA_LAUNCH_BLOCKING=1 kern_prof -lv program.py
```

kernel level
---

#### requirements
1. nvprof

After finding the most time-consuming calls, you can inspect deeper details by using the `profiler` provided by pytorch. The built-in profiler gives you the detailed elapsed time on basic tensor operations (such as `Addmm`, `IndexSelect`).

When the tensor operation is specified, I highly recommend you to write a simple test script where only  the specific tensor operation is called.
If you are able to locate the most expensive function call, you may use the nvprof to check if the existing kernel provided by pytorch meets your special use case.

```bash
nvprof --analysis-metrics -o test.nvprof --print-gpu-trace python test.py 2>> nvprof.log
```
And you can also modify the cuda kernel codes and then build the pytorch from source.

##### tuning CUDA kernels
some refs:[CUDA performance guide] [CUDA C programming guide]
##### build pytorch from source
```bash
# make sure you uninstall the released version of pytorch
pip uninstall pytorch

# pull the source code from github
git clone https://github.com/pytorch/pytorch
# pull the submodule dependencies
git submodule update --init
# build from source
python setup.py build [develop]
```
The optional `develop` is to specify whether the python files are mapped from git repo to `PYTHONPATH` or just copied to.
