Build pytorch with cuda-aware MPI support
====
Advantages:
  - Low cpu usage
  - low latency
  - no extra copy from/to system memory
  - easy to use

Requirements:
  - UVA support (available since )

Build mpi with CUDA support (openmpi)
---

dependencies:
  - automake
  - flex
  
```bash
apt install automake flex
```
> [color=#10d19a]**currently only openmpi-1.10 is supported by pytorch's compile system.**
> [name=Stone sky]
> [time=Thu, Apr 5, 2018 9:56 PM]
> 
> [color=#10d19a]openmpi-3.1.1 can compile with pytorch mater branch
> 
> [name=Stone sky]
> [time=Wed, Jul 25, 2018 11:39 PM]

download the source file from internet, you may refer to [up-to-date page](https://www.open-mpi.org/software/ompi/v3.0/)
```bash
wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.3.tar.gz # TODO: change to URL of 1.10.7
tar xvf openmpi-1.10.7.tar.gz
```

The newest release at this time is *3.0.1*, but I failed to build on that due to a building bug.
build from source with CUDA support.


```
cd openmpi-1.10.7
mkdir build && cd build
../configure --with-cuda --enable-mpi-thread-multiple # it's not tab completed by zsh
```
If your CUDA location is not `/usr/local/cuda` or you want to compile with non-default CUDA version, you may follow the [official-CUDA-tutorial](
https://www.open-mpi.org/faq/?category=buildcuda)  for customized build options.

Build pytorch with new open-mpi
----

1. build with system-wide mpi (older version)

Since the building system of pytorch looks for `libmpi` and `libmpicxx` at `/usr/lib`, while the default install path of open-mpi is `/usr/local/lib`. The general building process will raise error `mpi not found` for that. You can either copy/link the `.so` libraries or specify extra linking flags to compile successfully.

Workaround for pytorch
```bash
sudo cp /usr/local/lib/libmpi* /usr/lib
# compile pytorch
python setup.py clean
python setup.py build develop

# (optional, delete the redundant files)
sudo rm /usr/lib/libmpi*
```

Then export the libraries to `LD_LIBRARY_PATH` in case of `file not found` error.

```bash
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

2. build with arbitary version of mpi

Pytorch uses the *find_MPI* package bundled with *CMAKE*. In the newest CMAKE, it can automatically detect the MPI's lib and include path if an MPI compatible compiler is specified.

e.g.
```bash
python setup.py clean
CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) python setup.py build develop
```


How to check the CUDA-aware MPI support
---

1. simply list the dynamic libraries linked to pytorch's run-time.
`ldd torch/*.so`. If compiled with MPI, you can find `libmpi.so`. If compiled with CUDA-aware MPI, you can find `libopen-rte.so`.

2. run test-code


```python
import torch
import torch.distributed as dist

dist.init_process_group(backend='mpi')

t = torch.zeros(5,5).fill_(dist.get_rank()).cuda()

dist.all_reduce(t) # ???


```