# TE 矩阵相乘

- [Working with Operators Using Tensor Expression / Example 2: Manually Optimizing Matrix Multiplication with TE](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-2-manually-optimizing-matrix-multiplication-with-te)

代码：

- [te_mat_mul.py](../../frameworks/tvm/te_mat_mul/te_mat_mul.py)

结果：

```bash
$ cd frameworks/tvm/te_mat_mul/
$ python te_mat_mul.py
# Manually Optimizing Matrix Multiplication with TE

## Preparation and Performance Baseline
   target [llvm]: llvm -mcpu=core-avx2
Numpy running time: 0.007522
none: 1.453802
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x: int32, 0, 1024) {
    for (y: int32, 0, 1024) {
      C_2[((x*1024) + y)] = 0f32
      for (k: int32, 0, 1024) {
        let cse_var_2: int32 = (x*1024)
        let cse_var_1: int32 = (cse_var_2 + y)
        C_2[cse_var_1] = ((float32*)C_2[cse_var_1] + ((float32*)A_2[(cse_var_2 + k)]*(float32*)B_2[((k*1024) + y)]))
      }
    }
  }
}



## Optimization 1: Blocking
blocking: 0.214911
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        for (y.inner.init: int32, 0, 32) {
          C_2[((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)) + y.inner.init)] = 0f32
        }
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (x.inner: int32, 0, 32) {
            for (y.inner: int32, 0, 32) {
              let cse_var_3: int32 = (y.outer*32)
              let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
              let cse_var_1: int32 = ((cse_var_2 + cse_var_3) + y.inner)
              C_2[cse_var_1] = ((float32*)C_2[cse_var_1] + ((float32*)A_2[((cse_var_2 + (k.outer*4)) + k.inner)]*(float32*)B_2[((((k.outer*4096) + (k.inner*1024)) + cse_var_3) + y.inner)]))
            }
          }
        }
      }
    }
  }
}



## Optimization 2: Vectorization
vectorization: 0.234122
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        C_2[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (k.inner: int32, 0, 4) {
          for (x.inner: int32, 0, 32) {
            let cse_var_3: int32 = (y.outer*32)
            let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C_2[ramp(cse_var_1, 1, 32)] = ((float32x32*)C_2[ramp(cse_var_1, 1, 32)] + (broadcast((float32*)A_2[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*(float32x32*)B_2[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}



## Optimization 3: Loop Permutation
loop permutation: 0.084548
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner.init: int32, 0, 32) {
        C_2[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
      }
      for (k.outer: int32, 0, 256) {
        for (x.inner: int32, 0, 32) {
          for (k.inner: int32, 0, 4) {
            let cse_var_3: int32 = (y.outer*32)
            let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
            let cse_var_1: int32 = (cse_var_2 + cse_var_3)
            C_2[ramp(cse_var_1, 1, 32)] = ((float32x32*)C_2[ramp(cse_var_1, 1, 32)] + (broadcast((float32*)A_2[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*(float32x32*)B_2[ramp((((k.outer*4096) + (k.inner*1024)) + cse_var_3), 1, 32)]))
          }
        }
      }
    }
  }
}



## Optimization 4: Array Packing
array packing: 0.084020
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB[ramp(((x*32768) + (y*32)), 1, 32)] = (float32x32*)B_2[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) {
      for (y.outer: int32, 0, 32) {
        for (x.inner.init: int32, 0, 32) {
          C_2[ramp((((x.outer*32768) + (x.inner.init*1024)) + (y.outer*32)), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.inner: int32, 0, 32) {
            for (k.inner: int32, 0, 4) {
              let cse_var_2: int32 = ((x.outer*32768) + (x.inner*1024))
              let cse_var_1: int32 = (cse_var_2 + (y.outer*32))
              C_2[ramp(cse_var_1, 1, 32)] = ((float32x32*)C_2[ramp(cse_var_1, 1, 32)] + (broadcast((float32*)A_2[((cse_var_2 + (k.outer*4)) + k.inner)], 32)*(float32x32*)packedB[ramp((((y.outer*32768) + (k.outer*128)) + (k.inner*32)), 1, 32)]))
            }
          }
        }
      }
    }
  }
}



## Optimization 5: Optimizing Block Writing Through Caching
block caching: 0.077377
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global;
  allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB[ramp(((x*32768) + (y*32)), 1, 32)] = (float32x32*)B_2[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) {
      for (y.outer: int32, 0, 32) {
        for (x.c.init: int32, 0, 32) {
          C.global[ramp((x.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.c: int32, 0, 32) {
            let cse_var_3: int32 = (x.c*32)
            let cse_var_2: int32 = ((y.outer*32768) + (k.outer*128))
            let cse_var_1: int32 = (((x.outer*32768) + (x.c*1024)) + (k.outer*4))
             {
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[cse_var_1], 32)*(float32x32*)packedB[ramp(cse_var_2, 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 1)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 32), 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 2)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 64), 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 3)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 96), 1, 32)]))
            }
          }
        }
        for (x.inner: int32, 0, 32) {
          for (y.inner: int32, 0, 32) {
            C_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = (float32*)C.global[((x.inner*32) + y.inner)]
          }
        }
      }
    }
  }
}



## Optimization 6: Parallelization
parallelization: 0.020270
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  allocate(packedB: Pointer(global float32x32), float32x32, [32768]), storage_scope = global {
    for (x: int32, 0, 32) "parallel" {
      for (y: int32, 0, 1024) {
        packedB[ramp(((x*32768) + (y*32)), 1, 32)] = (float32x32*)B_2[ramp(((y*1024) + (x*32)), 1, 32)]
      }
    }
    for (x.outer: int32, 0, 32) "parallel" {
      allocate(C.global: Pointer(global float32), float32, [1024]), storage_scope = global;
      for (y.outer: int32, 0, 32) {
        for (x.c.init: int32, 0, 32) {
          C.global[ramp((x.c.init*32), 1, 32)] = broadcast(0f32, 32)
        }
        for (k.outer: int32, 0, 256) {
          for (x.c: int32, 0, 32) {
            let cse_var_3: int32 = (x.c*32)
            let cse_var_2: int32 = ((y.outer*32768) + (k.outer*128))
            let cse_var_1: int32 = (((x.outer*32768) + (x.c*1024)) + (k.outer*4))
             {
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[cse_var_1], 32)*(float32x32*)packedB[ramp(cse_var_2, 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 1)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 32), 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 2)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 64), 1, 32)]))
              C.global[ramp(cse_var_3, 1, 32)] = ((float32x32*)C.global[ramp(cse_var_3, 1, 32)] + (broadcast((float32*)A_2[(cse_var_1 + 3)], 32)*(float32x32*)packedB[ramp((cse_var_2 + 96), 1, 32)]))
            }
          }
        }
        for (x.inner: int32, 0, 32) {
          for (y.inner: int32, 0, 32) {
            C_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = (float32*)C.global[((x.inner*32) + y.inner)]
          }
        }
      }
    }
  }
}



## Summary of Matrix Multiplication Example
            Operator                  Timing             Performance
                none            1.4538019905                     1.0
            blocking     0.21491077570000003     0.14782671719006704
       vectorization            0.2341220135     0.16104119751513024
    loop permutation            0.0845482564     0.05815665197357563
       array packing            0.0840196047     0.05779301806506916
       block caching            0.0773771843      0.0532240186804174
     parallelization    0.020269596600000002    0.013942474100636475
```
