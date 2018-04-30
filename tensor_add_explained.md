Figure out what `b.add_(b)` has done
===
> This is a post about how to find the corresponding codes of a **Tensor** method, of the corresponding function `torch.add`

Why do I dive into the `Tensor.add_()`
---

These days I have been investigating the GFlops gap between theoretical PEAK and achieved of pytorch's `add` method. In brief, the `add` method performs element-wise addition between two **Tensors**.
On my test-bed, equipped with 2 Xeon E5 2620v4 which can achieve 256 GFlops PEAK performance according to Intel, the `b.add_(b)` (inplace add self) can only achieves 6GFlops at best.

I've tried to inspect the underlying code path for a simple matrix add.
Finally after a hard day, I found that for such **level-1 blas** ($O(N)$) operation, the memory bandwidth should definitely be the bottleneck of overall performance, no matter what advanced **SIMD** instructions (*AVX*, *SSE*, *NEON*) are utilized. Then the real **6** GFlops makes sense.

> bandwidth of current fastest DDR4 memory: 25.6 GB/s (X2 for load/save) [wiki-pedia](https://en.wikipedia.org/wiki/DDR4_SDRAM)
> load single precision floats: $\frac{25.6}{4}=6.4 GFlops$
> for non-inplace addition, which needs to load 2 floats for an addition, the Flops is halved.


How to find the codes into *AVX2* instructions
---

### Python binding and ATEN binding
For a comprehensive study on the auto-generation build system, please refer to blog *dispatch*.
ATEN is a `C/C++` Tensor library inspired by the need of pytorch, it is OO designed and also serves as the backend of pytorch's Python API.

Actually the Python API simply does some `Python/C` type conversion and error checking, plus invoking the corresponding method of ATEN **Tensor**.

Python binding and ATEN lib are generated at bulid time.
Since ATEN is still under actively develop, currently pytorch uses two scheme concurrently for automatic code generation, *native* and *cwrap*.

The declaration of `add` in cwrap file:
```
[[
  name: add
  variants:
    - method
    - function
  return: argument 0
  options:
    - cname: add_scaled
      arguments:
        - arg: THTensor* result
          output: True
        - THTensor* self
        - real other
        - arg: real alpha
          default: AS_REAL(1)
          kwarg_only: True
    - cname: cadd
      aten_sparse: True
      arguments:
        - arg: THTensor* result
          output: True
        - arg: THTensor* self
          broadcast: other fallback
        - arg: real alpha
          default: AS_REAL(1)
          kwarg_only: True
        - THTensor* other
    - sparse: True
      cname: spcadd
      aten_dense_sparse: True
      arguments:
        - arg: THTensor* result
          output: True
        - THTensor* self
        - arg: real alpha
          default: AS_REAL(1)
          kwarg_only: True
        - THSTensor* other
]]
```

As you see, the `Tensor.add` method may be dispatched into three different backend methods, based on the arguments. Specifically for `add`, `Tensor.add(5)` is going to call `add_scaled`, while `Tensor.add(Tensor)` will call `cadd`, if the Tensor is sparse, `spcadd` should be invoked.

Here because `add_` is simply an inplace alias of `add` which the first argument serves as both input Tensor and output Tensor. So we can simply find the corresponding backend name for `b.add_(b)`, **cadd**.

### Find the backend code of cadd method

Then you may search for the definition of cadd in TH(CPU), THC(GPU), THS(sparse). Because the actually name may be wrapped by pytorch's `THTensor_()` or `THCUDATensor_()` prefix, you have to figure it out depends on your backend. Since I'm looking for codes on CPU, so the wanted signature is `void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
`, which locates in `aten/src/TH/generic/THTensorMath.c`.

```clike=
void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int64_t srcSize = THTensor_(nElement)(src);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int srcContig = THTensor_(isContiguous)(src);
  int serial_path = 0;
  if (srcSize == r_Size){
    if (r_Contig && tContig && srcContig) {
      if(r_ == t) {
        THBlas_(axpy)(THTensor_(nElement)(t), value, THTensor_(data)(src), 1, THTensor_(data)(r_), 1);
      } else {
        TH_TENSOR_APPLY3_CONTIG(real, r_, real, t, real, src, THVector_(cadd)(r__data, t_data, src_data, value, r__len););
      }
    else // Non-contiguous case
```

The `THBlas(axpy)` calls standard blas function. Since I don't want to debug with the blas lib source code, I force the code to run line *15* by inserting a 0 into line *12*.

### TH_TENSOR_APPLY3_CONTIG
The `TH_TENSOR_APPLY3_CONTIG` is a macro to apply element-wise operation. It has two versions based on whether *OPENMP* support is enabled at compile time. When the size of *Tensor* is larger than the threshold (`TH_OMP_OVERHEAD_THREASHOLD`), the *Tensor* is split into `OMP_NUM_THREADS` parts, each processed by a thread.

```clike=
#ifdef _OPENMP
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = THTensor_(nElement)(TENSOR1); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = THTensor_(data)(TENSOR1) + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = THTensor_(data)(TENSOR2) + TH_TENSOR_offset; \
    TYPE3 *TENSOR3##_data = THTensor_(data)(TENSOR3) + TH_TENSOR_offset; \
    CODE \
  } \
}
```


### THVector_(cadd)

Since modern CPUs benefit from various SIMD(Single Instruction Multiple Data) instructions such as *SSE*, *AVX*, *NEON*, the backend instruction set of `THVector_(cadd)` is determined at **run time** by tranversing all the supported sets.

The `THVector_(cadd)` is dynamically defined via function pointer in `aten/src/TH/generic/THVectorDispatch.cpp`, with some useful macros defined in `aten/src/TH/generic/simd/simd.h`.

The codes for different SIMD is defined in `aten/src/TH/vector`. Since my CPU *Xeon E5 2620v4* supports *AVX2* which computes 256bit(8 single precision floats) in one instruction, I'm going to inspect the `AVX2.cpp`.

### THFloat_cadd_AVX2

Nothing special except for loop unrolling. Actually I didn't see any performance improve of such unrolling, because the major bottleneck is memory access, not the branching or prediction fails.

Unlike in GPU, the loop unrolling in CPU does not always gain speedup, because CPU is very good at sequential task and branching compared with GPU. Another reason is that the prediction fails at very low ratio when the tensor size goes up.

refer to [stackoverflowQA](https://stackoverflow.com/questions/24196076/is-gcc-loop-unrolling-flag-really-effective)
```clike=
void THFloatVector_cadd_AVX2(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  ptrdiff_t i;
  __m256 YMM15 = _mm256_set_ps(c, c, c, c, c, c, c, c);
  __m256 YMM0, YMM1, YMM2, YMM3;
  for (i=0; i<=((n)-16); i+=16) {
    YMM0 = _mm256_loadu_ps(y+i);
    YMM1 = _mm256_loadu_ps(y+i+8);
    YMM2 = _mm256_loadu_ps(x+i);
    YMM3 = _mm256_loadu_ps(x+i+8);
    YMM2 = _mm256_fmadd_ps(YMM0, YMM15, YMM2);
    YMM3 = _mm256_fmadd_ps(YMM1, YMM15, YMM3);
    _mm256_storeu_ps(z+i, YMM2);
    _mm256_storeu_ps(z+i+8, YMM3);
  }
  for (; i<(n); i++) {
    z[i] = x[i] + y[i] * c;
  }
}
```
