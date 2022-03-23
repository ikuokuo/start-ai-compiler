# TOPI 开始

> TOPI: TVM Operator Inventory

- [Introduction to TOPI](https://tvm.apache.org/docs/tutorial/intro_topi.html#sphx-glr-tutorial-intro-topi-py)

代码：

- [topi_start.py](../../frameworks/tvm/topi_start/topi_start.py)

结果：

```bash
$ cd frameworks/tvm/topi_start/
$ python topi_start.py
# Basic example
## compute the sum of rows
@main = primfn(A_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [n: int32, m: int32], [stride: int32, stride_1: int32], type="auto")}
  buffer_map = {A_1: A} {
  allocate(B: Pointer(global float32), float32, [n]), storage_scope = global;
  for (i: int32, 0, n) {
    B[i] = 0f32
    for (k: int32, 0, m) {
      B[i] = ((float32*)B[i] + (float32*)A_2[((i*stride) + (k*stride_1))])
    }
  }
}


## replace te.reduce_axis(), te.compute() with topi.sum()
@main = primfn(A_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [n: int32, m: int32], [stride: int32, stride_1: int32], type="auto")}
  buffer_map = {A_1: A} {
  allocate(A_red: Pointer(global float32), float32, [n]), storage_scope = global;
  for (ax0: int32, 0, n) {
    A_red[ax0] = 0f32
    for (k1: int32, 0, m) {
      A_red[ax0] = ((float32*)A_red[ax0] + (float32*)A_2[((ax0*stride) + (k1*stride_1))])
    }
  }
}



# Numpy-style operator overloading
## topi.broadcast_add() using '+'

# Generic schedules and fusing operations (CUDA)
/home/john/Codes/Star/tvm/python/tvm/target/target.py:282: UserWarning: Try specifying cuda arch by adding 'arch=sm_xx' to your target.
  warnings.warn("Try specifying cuda arch by adding 'arch=sm_xx' to your target.")
@main = primfn(a_1: handle, b_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {b: Buffer(b_2: Pointer(float32), float32, [10, 10], []),
             a: Buffer(a_2: Pointer(float32), float32, [100, 10, 10], [])}
  buffer_map = {a_1: a, b_1: b} {
  allocate(T_divide_red: Pointer(global float32), float32, [1]), storage_scope = global;
  attr [IterVar(threadIdx.x: int32, [0:1024], "ThreadIndex", "threadIdx.x")] "thread_extent" = 1024;
  allocate(T_divide_red.rf: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local {
    T_divide_red.rf[0] = 0f32
    for (k0.k1.fused.k2.fused.outer: int32, 0, 10) {
      if @tir.likely((((((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x) < 10000) && (((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x) < 10000)) && (((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x) < 10000)), dtype=bool) {
        T_divide_red.rf[0] = ((float32*)T_divide_red.rf[0] + ((((float32*)a_2[((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x)] + (float32*)b_2[floormod(((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x), 100)]) + ((float32*)a_2[((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x)]*(float32*)b_2[floormod(((k0.k1.fused.k2.fused.outer*1024) + threadIdx.x), 100)]))*0.5f32))
      }
    }
    attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
    @tir.tvm_thread_allreduce(1u32, (float32*)T_divide_red.rf[0], True, reduce_temp0, threadIdx.x, dtype=handle)
    if (threadIdx.x == 0) {
      T_divide_red[0] = (float32*)reduce_temp0[0]
    }
  }
}


[stage(a, placeholder(a, 0x560b9bf88640)), stage(b, placeholder(b, 0x560b9bf886c0)), stage(T_add, compute(T_add, body=[(a[ax0, ax1, ax2] + b[ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=broadcast, attrs={})), stage(T_multiply, compute(T_multiply, body=[(a[ax0, ax1, ax2]*b[ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=broadcast, attrs={})), stage(T_elemwise_sum, compute(T_elemwise_sum, body=[(T_add[ax0, ax1, ax2] + T_multiply[ax0, ax1, ax2])], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=elemwise, attrs={})), stage(T_divide, compute(T_divide, body=[(T_elemwise_sum[ax0, ax1, ax2]/2f)], axis=[iter_var(ax0, range(min=0, ext=100)), iter_var(ax1, range(min=0, ext=10)), iter_var(ax2, range(min=0, ext=10))], reduce_axis=[], tag=elemwise, attrs={})), stage(T_divide_red.rf, compute(T_divide_red.rf, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[T_divide[floordiv(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10), floormod(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10), floormod((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10)]], init=[], axis=[iter_var(k0.k1.fused.k2.fused.outer, range(min=0, ext=10))], where=tir.likely((((floordiv(floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10), 10) < 100) && (floordiv((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)), 10) < 1000)) && ((k0.k1.fused.k2.fused.inner + (k0.k1.fused.k2.fused.outer*1024)) < 10000))), value_index=0)], axis=[iter_var(k0.k1.fused.k2.fused.inner, range(min=0, ext=1024))], reduce_axis=[iter_var(k0.k1.fused.k2.fused.outer, range(min=0, ext=10))], tag=, attrs={})), stage(T_divide_red, compute(T_divide_red.repl, body=[reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]), source=[T_divide_red.rf[k0.k1.fused.k2.fused.inner.v]], init=[], axis=[iter_var(k0.k1.fused.k2.fused.inner.v, range(min=0, ext=1024))], where=(bool)1, value_index=0)], axis=[], reduce_axis=[iter_var(k0.k1.fused.k2.fused.inner.v, range(min=0, ext=1024))], tag=, attrs={}))]
## test the correctness
## schedule softmax
@main = primfn(tarray_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {tarray: Buffer(tarray_2: Pointer(float32), float32, [512, 512], [])}
  buffer_map = {tarray_1: tarray} {
  allocate(T_softmax_norm: Pointer(global float32x4), float32x4, [65536]), storage_scope = global;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 512;
  allocate(normal_reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(T_softmax_exp: Pointer(warp float32), float32, [512]), storage_scope = warp;
  allocate(normal_reduce_temp0_1: Pointer(local float32), float32, [1]), storage_scope = local;
  allocate(reduce_temp0_1: Pointer(local float32), float32, [1]), storage_scope = local {
    attr [IterVar(threadIdx.x: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
      normal_reduce_temp0[0] = -3.40282e+38f32
      for (k.inner: int32, 0, 16) {
        normal_reduce_temp0[0] = max((float32*)normal_reduce_temp0[0], (float32*)tarray_2[(((blockIdx.x*512) + (threadIdx.x*16)) + k.inner)])
      }
      attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, (float32*)normal_reduce_temp0[0], True, reduce_temp0, threadIdx.x, dtype=handle)
      for (i1.inner.outer: int32, 0, 4) {
        let cse_var_1: int32 = (i1.inner.outer*4)
        T_softmax_exp[ramp(((threadIdx.x*16) + cse_var_1), 1, 4)] = @tir.exp(((float32x4*)tarray_2[ramp((((blockIdx.x*512) + (threadIdx.x*16)) + cse_var_1), 1, 4)] - broadcast((float32*)reduce_temp0[0], 4)), dtype=float32x4)
      }
    }
    attr [IterVar(threadIdx.x, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
      normal_reduce_temp0_1[0] = 0f32
      for (k.inner_1: int32, 0, 16) {
        normal_reduce_temp0_1[0] = ((float32*)normal_reduce_temp0_1[0] + (float32*)T_softmax_exp[((threadIdx.x*16) + k.inner_1)])
      }
      attr [meta[tir.CommReducer][1]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
      @tir.tvm_thread_allreduce(1u32, (float32*)normal_reduce_temp0_1[0], True, reduce_temp0_1, threadIdx.x, dtype=handle)
      for (i1.inner.outer_1: int32, 0, 4) {
        let cse_var_2: int32 = (i1.inner.outer_1*4)
        T_softmax_norm[ramp((((blockIdx.x*512) + (threadIdx.x*16)) + cse_var_2), 1, 4)] = ((float32x4*)T_softmax_exp[ramp(((threadIdx.x*16) + cse_var_2), 1, 4)] / broadcast((float32*)reduce_temp0_1[0], 4))
      }
    }
  }
}



# Fusing convolutions (CUDA)
## fuse topi.nn.conv2d and topi.nn.relu together
Cannot find config for target=cuda -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -thread_warp_size=32, workload=('conv2d_nchw.cuda', ('TENSOR', (1, 3, 224, 224), 'float32'), ('TENSOR', (10, 3, 5, 5), 'float32'), 1, 2, 1). A fallback configuration is used, which may bring great performance regression.
@main = primfn(placeholder_2: handle, placeholder_3: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {placeholder_1: Buffer(placeholder_4: Pointer(float32), float32, [10, 3, 5, 5], []),
             placeholder: Buffer(placeholder_5: Pointer(float32), float32, [1, 3, 224, 224], [])}
  buffer_map = {placeholder_2: placeholder, placeholder_3: placeholder_1} {
  allocate(compute: Pointer(global float32), float32, [501760]), storage_scope = global;
  attr [IterVar(blockIdx.z: int32, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 5;
  allocate(conv2d_nchw: Pointer(local float32), float32, [14]), storage_scope = local;
  allocate(pad_temp.shared: Pointer(shared float32), float32, [112]), storage_scope = shared;
  allocate(placeholder.shared: Pointer(shared float32), float32, [2]), storage_scope = shared;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 224;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
    conv2d_nchw[0] = 0f32
    conv2d_nchw[2] = 0f32
    conv2d_nchw[4] = 0f32
    conv2d_nchw[6] = 0f32
    conv2d_nchw[8] = 0f32
    conv2d_nchw[10] = 0f32
    conv2d_nchw[12] = 0f32
    conv2d_nchw[1] = 0f32
    conv2d_nchw[3] = 0f32
    conv2d_nchw[5] = 0f32
    conv2d_nchw[7] = 0f32
    conv2d_nchw[9] = 0f32
    conv2d_nchw[11] = 0f32
    conv2d_nchw[13] = 0f32
    for (rc.outer: int32, 0, 3) {
      for (ry.outer: int32, 0, 5) {
        attr [IterVar(threadIdx.z_1: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared[(threadIdx.x_1*7)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (2 <= ((blockIdx.x*112) + (threadIdx.x_1*7)))), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 450)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 1)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (1 <= ((blockIdx.x*112) + (threadIdx.x_1*7)))), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 449)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared[threadIdx.x_2] = (float32*)placeholder_4[((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5))]
        }
        conv2d_nchw[0] = ((float32*)conv2d_nchw[0] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[0]))
        conv2d_nchw[2] = ((float32*)conv2d_nchw[2] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[4] = ((float32*)conv2d_nchw[4] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[6] = ((float32*)conv2d_nchw[6] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[8] = ((float32*)conv2d_nchw[8] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[10] = ((float32*)conv2d_nchw[10] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[12] = ((float32*)conv2d_nchw[12] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[1] = ((float32*)conv2d_nchw[1] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[1]))
        conv2d_nchw[3] = ((float32*)conv2d_nchw[3] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[5] = ((float32*)conv2d_nchw[5] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[7] = ((float32*)conv2d_nchw[7] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[9] = ((float32*)conv2d_nchw[9] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[11] = ((float32*)conv2d_nchw[11] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[13] = ((float32*)conv2d_nchw[13] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared[(threadIdx.x_1*7)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (1 <= ((blockIdx.x*112) + (threadIdx.x_1*7)))), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 449)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared[threadIdx.x_2] = (float32*)placeholder_4[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 1)]
        }
        conv2d_nchw[0] = ((float32*)conv2d_nchw[0] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[0]))
        conv2d_nchw[2] = ((float32*)conv2d_nchw[2] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[4] = ((float32*)conv2d_nchw[4] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[6] = ((float32*)conv2d_nchw[6] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[8] = ((float32*)conv2d_nchw[8] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[10] = ((float32*)conv2d_nchw[10] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[12] = ((float32*)conv2d_nchw[12] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[1] = ((float32*)conv2d_nchw[1] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[1]))
        conv2d_nchw[3] = ((float32*)conv2d_nchw[3] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[5] = ((float32*)conv2d_nchw[5] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[7] = ((float32*)conv2d_nchw[7] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[9] = ((float32*)conv2d_nchw[9] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[11] = ((float32*)conv2d_nchw[11] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[13] = ((float32*)conv2d_nchw[13] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 448)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 6)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared[threadIdx.x_2] = (float32*)placeholder_4[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 2)]
        }
        conv2d_nchw[0] = ((float32*)conv2d_nchw[0] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[0]))
        conv2d_nchw[2] = ((float32*)conv2d_nchw[2] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[4] = ((float32*)conv2d_nchw[4] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[6] = ((float32*)conv2d_nchw[6] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[8] = ((float32*)conv2d_nchw[8] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[10] = ((float32*)conv2d_nchw[10] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[12] = ((float32*)conv2d_nchw[12] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[1] = ((float32*)conv2d_nchw[1] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[1]))
        conv2d_nchw[3] = ((float32*)conv2d_nchw[3] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[5] = ((float32*)conv2d_nchw[5] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[7] = ((float32*)conv2d_nchw[7] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[9] = ((float32*)conv2d_nchw[9] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[11] = ((float32*)conv2d_nchw[11] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[13] = ((float32*)conv2d_nchw[13] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 447)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 5)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 6)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*112) + (threadIdx.x_1*7)) < 217)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 441)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared[threadIdx.x_2] = (float32*)placeholder_4[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 3)]
        }
        conv2d_nchw[0] = ((float32*)conv2d_nchw[0] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[0]))
        conv2d_nchw[2] = ((float32*)conv2d_nchw[2] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[4] = ((float32*)conv2d_nchw[4] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[6] = ((float32*)conv2d_nchw[6] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[8] = ((float32*)conv2d_nchw[8] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[10] = ((float32*)conv2d_nchw[10] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[12] = ((float32*)conv2d_nchw[12] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[1] = ((float32*)conv2d_nchw[1] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[1]))
        conv2d_nchw[3] = ((float32*)conv2d_nchw[3] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[5] = ((float32*)conv2d_nchw[5] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[7] = ((float32*)conv2d_nchw[7] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[9] = ((float32*)conv2d_nchw[9] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[11] = ((float32*)conv2d_nchw[11] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[13] = ((float32*)conv2d_nchw[13] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[1]))
        attr [IterVar(threadIdx.z_1, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_1, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_1, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
          pad_temp.shared[(threadIdx.x_1*7)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 446)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 1)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 445)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 2)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 444)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 3)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 443)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 4)] = @tir.if_then_else(((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 442)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 5)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*112) + (threadIdx.x_1*7)) < 217)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 441)], 0f32, dtype=float32)
          pad_temp.shared[((threadIdx.x_1*7) + 6)] = @tir.if_then_else((((2 <= (blockIdx.y + ry.outer)) && ((blockIdx.y + ry.outer) < 226)) && (((blockIdx.x*112) + (threadIdx.x_1*7)) < 216)), (float32*)placeholder_5[((((((rc.outer*50176) + (blockIdx.y*224)) + (ry.outer*224)) + (blockIdx.x*112)) + (threadIdx.x_1*7)) - 440)], 0f32, dtype=float32)
        }
        attr [IterVar(threadIdx.z_2, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
        attr [IterVar(threadIdx.y_2, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
        attr [IterVar(threadIdx.x_2, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
        if @tir.likely((threadIdx.x_2 < 2), dtype=bool) {
          placeholder.shared[threadIdx.x_2] = (float32*)placeholder_4[(((((blockIdx.z*150) + (threadIdx.x_2*75)) + (rc.outer*25)) + (ry.outer*5)) + 4)]
        }
        conv2d_nchw[0] = ((float32*)conv2d_nchw[0] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[0]))
        conv2d_nchw[2] = ((float32*)conv2d_nchw[2] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[4] = ((float32*)conv2d_nchw[4] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[6] = ((float32*)conv2d_nchw[6] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[8] = ((float32*)conv2d_nchw[8] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[10] = ((float32*)conv2d_nchw[10] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[12] = ((float32*)conv2d_nchw[12] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[0]))
        conv2d_nchw[1] = ((float32*)conv2d_nchw[1] + ((float32*)pad_temp.shared[threadIdx.x]*(float32*)placeholder.shared[1]))
        conv2d_nchw[3] = ((float32*)conv2d_nchw[3] + ((float32*)pad_temp.shared[(threadIdx.x + 16)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[5] = ((float32*)conv2d_nchw[5] + ((float32*)pad_temp.shared[(threadIdx.x + 32)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[7] = ((float32*)conv2d_nchw[7] + ((float32*)pad_temp.shared[(threadIdx.x + 48)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[9] = ((float32*)conv2d_nchw[9] + ((float32*)pad_temp.shared[(threadIdx.x + 64)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[11] = ((float32*)conv2d_nchw[11] + ((float32*)pad_temp.shared[(threadIdx.x + 80)]*(float32*)placeholder.shared[1]))
        conv2d_nchw[13] = ((float32*)conv2d_nchw[13] + ((float32*)pad_temp.shared[(threadIdx.x + 96)]*(float32*)placeholder.shared[1]))
      }
    }
    compute[((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x)] = max((float32*)conv2d_nchw[0], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 16)] = max((float32*)conv2d_nchw[2], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 32)] = max((float32*)conv2d_nchw[4], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 48)] = max((float32*)conv2d_nchw[6], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 64)] = max((float32*)conv2d_nchw[8], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 80)] = max((float32*)conv2d_nchw[10], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 96)] = max((float32*)conv2d_nchw[12], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50176)] = max((float32*)conv2d_nchw[1], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50192)] = max((float32*)conv2d_nchw[3], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50208)] = max((float32*)conv2d_nchw[5], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50224)] = max((float32*)conv2d_nchw[7], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50240)] = max((float32*)conv2d_nchw[9], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50256)] = max((float32*)conv2d_nchw[11], 0f32)
    compute[(((((blockIdx.z*100352) + (blockIdx.y*224)) + (blockIdx.x*112)) + threadIdx.x) + 50272)] = max((float32*)conv2d_nchw[13], 0f32)
  }
}
```
