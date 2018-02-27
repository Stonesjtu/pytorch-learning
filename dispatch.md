# how pytorch dispatch the `Variable.index_select` and the auto-grad

This post assumes readers have the basic concepts of the `Function` and `Variable` in `pytorch`.

pytorch is a deep learning framework that is famous for the simplicity of prototyping and debugging. There are no separate graph defining and actual computing parts in pytorch, instead, it builds the computation graph on-the-fly.

## The `C/C++` backend

For pytorch, or any other deep learning frameworks such as `mxnet`, `tensorflow`, the backend is usually written in `C/C++` for best performance. It does no more than maintaining tensor information and doing tensor(matrix) math.

The backend of pytorch is somehow fragile because it re-uses many codes from project `torch`. Fortunately the pytorch community is working on a new unified tensor framework named `ATEN` which is already used since version `0.3.0`. At the current stage, lib `ATEN` defines data structures such as `Tensor`, `Storage`, `TensorInfo` ...

Currently the computation is dispatched from `ATEN` to the corresponding methods defined in `TH` (CPU), `THC` (GPU) and perhaps `THS` (sparse matrix math). Below the more complicated `THC` backend is used to clarify the function path from `ATEN` to real computing kernels.

### invoking CUDA kernels

- kernel wrappers
The wrappers perform some trivial tasks before and after kernel launching, such as error checking, data preparation and setting the kernel runtime parameters, e.g. the `blockSize`,`gridSize`, `stream ID`. 
```C
// pytorch/aten/src/THC/generic/THCTensorIndex.cu
void THCTensor_(gather)(THCState* state, THCTensor *tensor,
                         THCTensor *src, int dim, THCudaLongTensor *index) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(THCudaLongTensor_nDimension(state, index) == THCTensor_(nDimension)(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THLongStorage *indexSize = THCudaLongTensor_newSizeOf(state, index);
  THArgCheck(THCTensor_(isSize)(state, tensor, indexSize), 4,
             "Index tensor must have the same size as output tensor.");
  THLongStorage_free(indexSize);
  // to invoke CUDA kernel
```

- CUDA kernels
The CUDA kernels are codes running on CUDA-compatible GPUs, they are generally the critical part to optmize.
```C
// pytorch/aten/src/THC/THCTensorIndex.cu
template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_gatherKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < src.sizes[dim]);
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}
```


## Binding `Python` and `C/C++` API
Pytorch uses `cwrap` to declare the binding of python call and its' corresponding C backend. Simply searching `.cwrap` file in pytorch repo will give you some hints.

In detail, `cwrap` files are used to generate `PyMethods` for python type object `torch.Tensor` at building phase (when running `python setup.py build`). `PyMethod` is the inner mechanism of Cpython's object implementation, it enables python object to call native C functions, the ATEN function in Pytorch. After defining the proper `PyMethod`, the `Variable.index_select` call in Python finally invokes the corresponding C backend function `THCTensor_(indexSelect)`.

See docs:
`pytorch/aten/src/ATen/native/README.md`

And source codes:
`pytorch/aten/src/ATen/Declarations.cwrap`

## Defining the `forward`/`backward` pair
In early version, pytorch uses hand-written `forward` and `backward` methods in each `Function` class. But now a separate declaration file  is employed to define the `forward`/`backward` pair for simplicity.

Like the `Python/C binding`, the declaration file is also used to generate `Variable` methods at building stage.
e.g. for `Variable.index_select`, the gradient of self is obtained by calling the `grad.type().zeros(self.sizes()).index_add_(dim, index, grad)`.

code snippets for `forward`/`backward` binding:
```yaml
# pytorch/tools/autograd/derivatives.yaml
- name: index_select(Tensor self, int64_t dim, Tensor index)
  self: grad.type().zeros(self.sizes()).index_add_(dim, index, grad)

- name: kthvalue(Tensor self, int64_t k, int64_t dim, bool keepdim)
  self: select_backward(grad, dim, indices, self.sizes(), keepdim)
```
Here I extract some useful comments from the `derivatives.yaml`:
> Each entry consists of:
> - A 'name', which specifies the ATen name of the function you
>   are defining derivatives for, and an argument specification.
> - One or more gradients entries, mapping a differentiable input
>   names to a formula specifying how to compute its gradient.
>   Note that a single gradient entry can specify the gradient
>   formula for multiple input names, by specifying a key
>   "self, other" (see atan2 for an example).
The values in this yaml file are standard `C++` (C++11 exactly) statements without trailing semi-colons, which will be invoked by `backward engine` to apply chain rule.
There are two approaches to defining the backward function, a simple one-liner or a more complex function defined in `pytorch/tools/autograd/templates/Functions.cpp`. For example, the backward function for `kthvalue()` is `select_backward()`,

```C
// pytorch/tools/autograd/templates/Functions.cpp
Tensor sum_backward(const Tensor & grad, IntList sizes, int64_t dim, bool keepdim) {
#ifdef WITH_SCALARS
  if (!keepdim && sizes.size() > 0) {
#else
  if (!keepdim && sizes.size() > 1) {
#endif
    return grad.unsqueeze(dim).expand(sizes);
  } else {
    return grad.expand(sizes);
  }
}
```

In the meanwhile, users can still write their own `Function` in Python by subclassing and defining their own `forward` and `backward` methods.
See source code:
`pytorch/tools/autograd/derivatives.yaml`
`pytorch/tools/autograd/templates/Functions.cpp`
