Build pytorch with NCCL2
=====

Pre
---

#### What is NCCL2?
Check this out:
https://github.com/PaddlePaddle/Paddle/wiki/NCCL2-Survey

Check your NCCL version:
    - pytorch comes with NCCL 1 bundled, but it will detect the system-wide NCCL for better performance (**Compile time only**).
    - `vim /usr/include/nccl.h` to check the NCCL version.

Step by Step
---

1. Download NCCL2 runtime and header files.
NCCL2 is not open-sourced, so you have to download the compiled version from NVIDIA's website. Or directly from the repo file server: http://developer.download.nvidia.com/compute/machine-learning/repos
warning: you have to choose the right system version and hardware architecture.
Choose the right NCCL version and cuda version. Here I downloaded `libnccl2_2.2.12-1+cuda8.0_amd64.deb` and `libnccl-dev_2.2.12-1+cuda8.0_amd64.deb`.


2. Install NCCL2:
It's quite straightforward: (ubuntu-16.04LTS for e.g.)
```bash
sudo dpkg -i libnccl2_2.2.12-1+cuda8.0_amd64.deb libnccl-dev_2.2.12-1+cuda8.0_amd64.deb
```
> I suffered to find the location of installed files (header and lib), simply run `dpkg -c *.deb` solves the problem.


2. compile pytorch from source
After the installation of new NCCL, you have to build the pytorch from a clean directory.
```bash
cd pytorch

# both of the following two lines are required, or the cached NCCL1 will
# be used instead of NCCL2
rm build -rf
rm torch/lib/tmp_install -rf
# or you can run this for a complete clean directory:
# python setup.py build clean

python setup.py build develop
```

If your NCCL2 is installed into customized directory, you can pass the location by environment variables. (haven't tested by myself)
```
NCCL_LIB_DIR=~/.local/lib NCCL_INCLUDE_DIR=/~/.local/include python setup.py build develop
```

3. run

test_script

> on node-0

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file://distributed_test",
                        world_size=2,
                        rank=0)
print('haha')
tensor_list = []
for dev_idx in range(2):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)
```

> on node-1

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file://distributed_test",
                        world_size=2,
                        rank=1)
print('haha')
tensor_list = []
for dev_idx in range(2):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)
```

#### how to run:

- **critical**: choose the right NIC for inter-node communication

Sometimes NCCL for inter-node communication fails to setup connections with each other, then it can raise error *unhandled system error*. In my system it's like
```log
RuntimeError: NCCL error in: /slwork/users/kys10/Workspace/pytorch/torch/lib/THD/base/data_channels/DataChannelNccl.cpp:322, unhandled system error
```
Actually if you runs the inter-node version on NCCL1 which only supports intra-node communication, the error message is alwo `unhandled system error`

In this case you have to specify the NIC viable for inter-node connection.

Typically, in the machine with docker installed, the error can be workaround by:
`NCCL_SOCKET_IFNAME=^docker0 python node-0.py`

ref:
- [nvidia-forum](https://devtalk.nvidia.com/default/topic/1023946/gpu-accelerated-libraries/nccl-2-0-support-inter-node-communication-using-sockets-/)
- [NCCL2 doc](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs)
