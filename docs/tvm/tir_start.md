# TensorIR 开始

- [Blitz Course to TensorIR](https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html#sphx-glr-tutorial-tensor-ir-blitz-course-py)

代码：

- [tir_start.py](../../frameworks/tvm/tir_start/tir_start.py)

结果：

```bash
$ cd frameworks/tvm/tir_start/
$ python tir_start.py
# Create an IRModule
<class 'tvm.ir.module.IRModule'>
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i in T.serial(8):
            with T.block("B"):
                vi = T.axis.spatial(8, i)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1)



## TE to IRModule
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0 in T.serial(8):
            with T.block("B"):
                i0_1 = T.axis.spatial(8, i0)
                T.reads(A[i0_1])
                T.writes(B[i0_1])
                B[i0_1] = A[i0_1] + T.float32(1)



# Build and Run an IRModule
<class 'tvm.driver.build_module.OperatorModule'>
[0. 1. 2. 3. 4. 5. 6. 7.]
[1. 2. 3. 4. 5. 6. 7. 8.]

# Transform an IRModule
<class 'tvm.tir.schedule.schedule.Schedule'>
## tile the loop into 3 loops
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i_0, i_1, i_2 in T.grid(2, 2, 2):
            with T.block("B"):
                vi = T.axis.spatial(8, i_0 * 4 + i_1 * 2 + i_2)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1)


## reorder the loops, move loop i_2 to outside of i_1
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i_0, i_2, i_1 in T.grid(2, 2, 2):
            with T.block("B"):
                vi = T.axis.spatial(8, i_0 * 4 + i_1 * 2 + i_2)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1)



# Transform to a GPU program [Y/n]
## binding the threads
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(8,), "float32"], B: T.Buffer[(8,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i_0 in T.thread_binding(2, thread="blockIdx.x"):
            for i_2 in T.thread_binding(2, thread="threadIdx.x"):
                for i_1 in T.serial(2):
                    with T.block("B"):
                        vi = T.axis.spatial(8, i_0 * 4 + i_1 * 2 + i_2)
                        T.reads(A[vi])
                        T.writes(B[vi])
                        B[vi] = A[vi] + T.float32(1)


## build the IRModule with cuda backends
[0. 1. 2. 3. 4. 5. 6. 7.]
[1. 2. 3. 4. 5. 6. 7. 8.]
```
