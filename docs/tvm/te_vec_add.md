# TE 向量相加

- [Working with Operators Using Tensor Expression / Example 1: Writing and Scheduling Vector Addition in TE for CPU](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-1-writing-and-scheduling-vector-addition-in-te-for-cpu)

<!--
$ llc --version | grep CPU
  Host CPU: skylake
$ lscpu
-->

代码：

- [te_vec_add.py](../../frameworks/tvm/te_vec_add/te_vec_add.py)

结果：

```bash
$ conda activate tvm
$ pip install pytest

$ cd frameworks/tvm/te_vec_add/
$ python te_vec_add.py
# Writing and Scheduling Vector Addition in TE for CPU
  target [llvm]: llvm -mcpu=core-avx2

## Describing the Vector Computation

## Create a Default Schedule for the Computation

## Compile and Evaluate the Default Schedule
   get a comparison of how fast this version is compared to numpy
Numpy running time: 0.000007
naive: 0.000006

## Updating the Schedule to Use Paralleism
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [n], [stride_1: int32], type="auto"),
             A: Buffer(A_2: Pointer(float32), float32, [n], [stride_2: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, n) "parallel" {
    C_2[(i*stride)] = ((float32*)A_2[(i*stride_2)] + (float32*)B_2[(i*stride_1)])
  }
}


parallel: 0.000002

## Updating the Schedule to Use Vectorization
vector: 0.000006
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [n], [stride_1: int32], type="auto"),
             A: Buffer(A_2: Pointer(float32), float32, [n], [stride_2: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer: int32, 0, floordiv((n + 3), 4)) "parallel" {
    for (i.inner.s: int32, 0, 4) {
      if @tir.likely((((i.outer*4) + i.inner.s) < n), dtype=bool) {
        let cse_var_1: int32 = ((i.outer*4) + i.inner.s)
        C_2[(cse_var_1*stride)] = ((float32*)A_2[(cse_var_1*stride_2)] + (float32*)B_2[(cse_var_1*stride_1)])
      }
    }
  }
}



## Comparing the Different Schedules
            Operator                  Timing             Performance
               numpy    7.047870312817395e-06                    1.0
               naive              6.2333e-06      0.8844231978366569
            parallel              2.0002e-06      0.2838020439113922
              vector              5.7546e-06       0.816501970749174

# ------------------------------------------------------------------------------
# Targeting Vector Addition for GPUs [Y/n]
  target options: cuda (NVIDIA GPUs), rocm (Radeon GPUS), OpenCL (opencl)
  target [cuda]:
<class 'tvm.te.tensor.Tensor'>
-----GPU code-----

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(64) myadd_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, int n, int stride, int stride1, int stride2) {
  if (((int)blockIdx.x) < (n >> 6)) {
    C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
    }
  }
}



# ------------------------------------------------------------------------------
# Saving and Loading Compiled Modules
  tempdir=/tmp/tmppzk4rb9i
  tgt.kind.name=cuda
['myadd.cubin', 'myadd.so', 'myadd.o', 'myadd.tvm_meta.json']
## Load Compiled Module
## Pack Everything into One Library
   currently we support packing of Metal, OpenCL and CUDA modules [Y/n]

# ------------------------------------------------------------------------------
# Generate OpenCL Code (target=opencl) [N]

# ------------------------------------------------------------------------------
TE Scheduling Primitives
  split: splits a specified axis into two axises by the defined factor.
  tile: tiles will split a computation across two axes by the defined factors.
  fuse: fuses two consecutive axises of one computation.
  reorder: can reorder the axises of a computation into a defined order.
  bind: can bind a computation to a specific thread, useful in GPU programming.
  compute_at: by default, TVM will compute tensors at the outermost level of the function, or the root, by default. compute_at specifies that one tensor should be computed at the first axis of computation for another operator.
  compute_inline: when marked inline, a computation will be expanded then inserted into the address where the tensor is required.
  compute_root: moves a computation to the outermost layer, or root, of the function. This means that stage of the computation will be fully computed before it moves on to the next stage.

Schedule Primitives
  https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html#schedule-primitives
```
