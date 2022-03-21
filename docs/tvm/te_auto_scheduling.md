# TE 自动调度

- [Optimizing Operators with Auto-scheduling](https://tvm.apache.org/docs/tutorial/auto_scheduler_matmul_x86.html#sphx-glr-tutorial-auto-scheduler-matmul-x86-py)

代码：

- [te_auto_scheduling.py](../../frameworks/tvm/te_auto_scheduling/te_auto_scheduling.py)

结果：

```bash
$ cd frameworks/tvm/te_auto_scheduling/
$ python te_auto_scheduling.py
# Defining the Matrix Multiplication

# Create the search task
  target [llvm]: llvm -mcpu=core-avx2
Computational DAG:
A = PLACEHOLDER [1024, 1024]
B = PLACEHOLDER [1024, 1024]
matmul(i, j) += (A[i, k]*B[k, j])
C = PLACEHOLDER [1024, 1024]
out(i, j) = (matmul[i, j] + C[i, j])


# Set Parameters for Auto-Scheduler

# Run the search
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Generate Sketches               #s: 3
Sample Initial Population       #s: 2024        fail_ct: 0      Time elapsed: 0.49
GA Iter: 0      Max score: 0.9995       Min score: 0.9382       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 4      Max score: 0.9995       Min score: 0.9841       #Pop: 128       #M+: 1375       #M-: 69
EvolutionarySearch              #s: 128 Time elapsed: 2.02
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 10 programs to measure:
..........T*********
==================================================
No: 1   GFLOPS: 8.10 / 8.10     results: MeasureResult(cost:[0.2653], error_no:0, all_cost:1.46, Tstamp:1647846117.91)
==================================================
Placeholder: A, B, C
matmul auto_unroll: 16
parallel i.0@j.0@i.1@j.1@ (0,8)
  for k.0 (0,16)
    for i.2 (0,32)
      for j.2 (0,32)
        for k.1 (0,64)
          for i.3 (0,32)
            vectorize j.3 (0,4)
              matmul = ...
parallel i (0,1024)
  for j (0,1024)
    out = ...

==================================================
No: 2   GFLOPS: 216.96 / 216.96 results: MeasureResult(cost:[0.0099], error_no:0, all_cost:0.88, Tstamp:1647846118.18)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@i.1@j.1@ (0,256)
  for k.0 (0,16)
    for i.2 (0,4)
      for j.2 (0,4)
        for k.1 (0,64)
          for i.3 (0,4)
            for j.3 (0,64)
              matmul = ...
  for i.2 (0,16)
    for j.2 (0,256)
      out = ...

==================================================
No: 3   GFLOPS: 29.48 / 216.96  results: MeasureResult(cost:[0.0729], error_no:0, all_cost:1.82, Tstamp:1647846118.55)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@ (0,32)
  matmul auto_unroll: 512
  for i.1 (0,4)
    for j.1 (0,128)
      for k.0 (0,128)
        for j.2 (0,4)
          for k.1 (0,8)
            for i.3 (0,16)
              matmul = ...
  for i.1 (0,64)
    for j.1 (0,512)
      out = ...

==================================================
No: 4   GFLOPS: 183.88 / 216.96 results: MeasureResult(cost:[0.0117], error_no:0, all_cost:0.37, Tstamp:1647846118.80)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@i.1@ (0,128)
  for k.0 (0,1024)
    for i.2 (0,64)
      for j.2 (0,16)
        vectorize j.3 (0,8)
          matmul = ...
  for i.2 (0,64)
    for j.2 (0,128)
      out = ...

==================================================
No: 5   GFLOPS: 12.73 / 216.96  results: MeasureResult(cost:[0.1687], error_no:0, all_cost:1.05, Tstamp:1647846119.57)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@i.1@ (0,1024)
  for j.1 (0,256)
    matmul auto_unroll: 64
    for k.0 (0,16)
      for i.2 (0,2)
        for j.2 (0,2)
          for k.1 (0,64)
            matmul = ...
    for i.2 (0,2)
      for j.2 (0,2)
        out = ...

==================================================
No: 6   GFLOPS: 0.00 / 216.96   results: MeasureResult(error_type:BuildTimeoutError, error_msg:, all_cost:15.00, Tstamp:1647846119.57)
==================================================
Placeholder: A, B, C
matmul auto_unroll: 512
parallel i.0@j.0@i.1@j.1@ (0,128)
  for k.0 (0,1024)
    for i.2 (0,16)
      for j.2 (0,8)
        for i.3 (0,2)
          for j.3 (0,32)
            matmul = ...
parallel i (0,1024)
  for j (0,1024)
    out = ...

==================================================
No: 7   GFLOPS: 118.09 / 216.96 results: MeasureResult(cost:[0.0182], error_no:0, all_cost:2.59, Tstamp:1647846119.83)
==================================================
Placeholder: A, B, C
matmul auto_unroll: 512
parallel i.0@j.0@i.1@ (0,512)
  for j.1 (0,2)
    for k.0 (0,256)
      for j.2 (0,4)
        for k.1 (0,4)
          for i.3 (0,8)
            for j.3 (0,32)
              matmul = ...
parallel i (0,1024)
  for j (0,1024)
    out = ...

==================================================
No: 8   GFLOPS: 3.07 / 216.96   results: MeasureResult(cost:[0.6990], error_no:0, all_cost:3.16, Tstamp:1647846122.71)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@ (0,128)
  for i.1 (0,2)
    for j.1 (0,4)
      for k.0 (0,512)
        for i.2 (0,2)
          for j.2 (0,16)
            for k.1 (0,2)
              for i.3 (0,32)
                matmul = ...
parallel i (0,1024)
  for j (0,1024)
    out = ...

==================================================
No: 9   GFLOPS: 118.05 / 216.96 results: MeasureResult(cost:[0.0182], error_no:0, all_cost:0.70, Tstamp:1647846122.99)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@ (0,512)
  matmul auto_unroll: 512
  for k.0 (0,64)
    for i.2 (0,8)
      for j.2 (0,32)
        for k.1 (0,16)
          for i.3 (0,8)
            matmul = ...
  for i.1 (0,64)
    for j.1 (0,32)
      out = ...

==================================================
No: 10  GFLOPS: 83.16 / 216.96  results: MeasureResult(cost:[0.0258], error_no:0, all_cost:0.74, Tstamp:1647846123.29)
==================================================
Placeholder: A, B, C
parallel i.0@j.0@ (0,1024)
  matmul auto_unroll: 16
  for i.1 (0,2)
    for j.1 (0,2)
      for k.0 (0,64)
        for i.2 (0,8)
          for j.2 (0,8)
            for k.1 (0,16)
              for i.3 (0,2)
                vectorize j.3 (0,2)
                  matmul = ...
  for i.1 (0,32)
    for j.1 (0,32)
      out = ...

Time elapsed for measurement: 22.22 s
----------------------------------------------------------------------
------------------------------  [ Done ]
----------------------------------------------------------------------

# Inspecting the Optimized Schedule
Lowered TIR:
@main = primfn(A_1: handle, B_1: handle, C_1: handle, out_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {out: Buffer(out_2: Pointer(float32), float32, [1024, 1024], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C, out_1: out} {
  allocate(auto_scheduler_layout_transform: Pointer(global float32), float32, [1048576]), storage_scope = global {
    for (ax0.ax1.fused.ax2.fused: int32, 0, 4) "parallel" {
      for (ax4: int32, 0, 16) {
        for (ax5: int32, 0, 4) {
          for (ax6: int32, 0, 64) {
            for (ax7: int32, 0, 64) {
              auto_scheduler_layout_transform[(((((ax0.ax1.fused.ax2.fused*262144) + (ax4*16384)) + (ax5*4096)) + (ax6*64)) + ax7)] = (float32*)B_2[(((((ax4*65536) + (ax6*1024)) + (ax0.ax1.fused.ax2.fused*256)) + (ax5*64)) + ax7)]
            }
          }
        }
      }
    }
    for (i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused: int32, 0, 256) "parallel" {
      allocate(matmul: Pointer(global float32), float32, [4096]), storage_scope = global {
        for (i.outer.inner.init: int32, 0, 4) {
          for (j.outer.inner.init: int32, 0, 4) {
            for (i.inner.init: int32, 0, 4) {
              for (j.inner.init: int32, 0, 64) {
                matmul[((((i.outer.inner.init*1024) + (i.inner.init*256)) + (j.outer.inner.init*64)) + j.inner.init)] = 0f32
              }
            }
          }
        }
        for (k.outer: int32, 0, 16) {
          for (i.outer.inner: int32, 0, 4) {
            for (j.outer.inner: int32, 0, 4) {
              for (k.inner: int32, 0, 64) {
                for (i.inner: int32, 0, 4) {
                  for (j.inner: int32, 0, 64) {
                    let cse_var_1: int32 = ((((i.outer.inner*1024) + (i.inner*256)) + (j.outer.inner*64)) + j.inner)
                    matmul[cse_var_1] = ((float32*)matmul[cse_var_1] + ((float32*)A_2[(((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*16384) + (i.outer.inner*4096)) + (i.inner*1024)) + (k.outer*64)) + k.inner)]*(float32*)auto_scheduler_layout_transform[(((((floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*262144) + (k.outer*16384)) + (j.outer.inner*4096)) + (k.inner*64)) + j.inner)]))
                  }
                }
              }
            }
          }
        }
        for (i.inner_1: int32, 0, 16) {
          for (j.inner_1: int32, 0, 256) {
            let cse_var_2: int32 = ((((floordiv(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*16384) + (i.inner_1*1024)) + (floormod(i.outer.outer.j.outer.outer.fused.i.outer.inner.fused.j.outer.inner.fused, 4)*256)) + j.inner_1)
            out_2[cse_var_2] = ((float32*)matmul[((i.inner_1*256) + j.inner_1)] + (float32*)C_2[cse_var_2])
          }
        }
      }
    }
  }
}



# Check correctness and evaluate performance
Execution time of this operator: 10.325 ms

# Using the record file
Equivalent python schedule:
matmul_i, matmul_j, matmul_k = tuple(matmul.op.axis) + tuple(matmul.op.reduce_axis)
out_i, out_j = tuple(out.op.axis) + tuple(out.op.reduce_axis)
matmul_i_o_i, matmul_i_i = s[matmul].split(matmul_i, factor=4)
matmul_i_o_o_i, matmul_i_o_i = s[matmul].split(matmul_i_o_i, factor=4)
matmul_i_o_o_o, matmul_i_o_o_i = s[matmul].split(matmul_i_o_o_i, factor=32)
matmul_j_o_i, matmul_j_i = s[matmul].split(matmul_j, factor=64)
matmul_j_o_o_i, matmul_j_o_i = s[matmul].split(matmul_j_o_i, factor=4)
matmul_j_o_o_o, matmul_j_o_o_i = s[matmul].split(matmul_j_o_o_i, factor=4)
matmul_k_o, matmul_k_i = s[matmul].split(matmul_k, factor=64)
s[matmul].reorder(matmul_i_o_o_o, matmul_j_o_o_o, matmul_i_o_o_i, matmul_j_o_o_i, matmul_k_o, matmul_i_o_i, matmul_j_o_i, matmul_k_i, matmul_i_i, matmul_j_i)
out_i_o_i, out_i_i = s[out].split(out_i, factor=16)
out_i_o_o, out_i_o_i = s[out].split(out_i_o_i, factor=32)
out_j_o_i, out_j_i = s[out].split(out_j, factor=256)
out_j_o_o, out_j_o_i = s[out].split(out_j_o_i, factor=4)
s[out].reorder(out_i_o_o, out_j_o_o, out_i_o_i, out_j_o_i, out_i_i, out_j_i)
s[matmul].compute_at(s[out], out_j_o_i)
out_i_o_o_j_o_o_fused_i_o_i_fused_j_o_i_fused = s[out].fuse(out_i_o_o, out_j_o_o, out_i_o_i, out_j_o_i)
s[out].parallel(out_i_o_o_j_o_o_fused_i_o_i_fused_j_o_i_fused)
s[matmul].pragma(matmul_i_o_o_o, "auto_unroll_max_step", 0)
s[matmul].pragma(matmul_i_o_o_o, "unroll_explicit", True)

Resume search:
/home/john/anaconda3/envs/tvm/lib/python3.8/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
----------------------------------------------------------------------
------------------------------  [ Call init-search callbacks ]
----------------------------------------------------------------------
SearchPolicy: Loaded 10 measurement records from matmul.json for ["matmul_add", 1024, 1024, 1024, "float32"]
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Generate Sketches               #s: 3
Sample Initial Population       #s: 2016        fail_ct: 1      Time elapsed: 0.58
GA Iter: 0      Max score: 0.9996       Min score: 0.9438       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 4      Max score: 1.0000       Min score: 0.9888       #Pop: 128       #M+: 1379       #M-: 74
EvolutionarySearch              #s: 128 Time elapsed: 2.30
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 5 programs to measure:
.....*****
Time elapsed for measurement: 2.46 s
----------------------------------------------------------------------
------------------------------  [ Done ]
----------------------------------------------------------------------
```
