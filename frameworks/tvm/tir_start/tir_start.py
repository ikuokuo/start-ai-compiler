#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np


print("# Create an IRModule")

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print(type(ir_module))
print(ir_module.script())


print("\n## TE to IRModule")

from tvm import te

A = te.placeholder((8,), dtype="float32", name="A")
B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")
func = te.create_prim_func([A, B])
ir_module_from_te = IRModule({"main": func})
print(ir_module_from_te.script())


print("\n# Build and Run an IRModule")

mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

a = tvm.nd.array(np.arange(8).astype("float32"))
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(a)
print(b)


print("\n# Transform an IRModule")

sch = tvm.tir.Schedule(ir_module)
print(type(sch))

print("## tile the loop into 3 loops")

# Get block by its name
block_b = sch.get_block("B")
# Get loops surronding the block
(i,) = sch.get_loops(block_b)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[2, 2, 2])
print(sch.mod.script())

print("## reorder the loops, move loop i_2 to outside of i_1")

sch.reorder(i_0, i_2, i_1)
print(sch.mod.script())


s = input("\n# Transform to a GPU program [Y/n] ")
if s and s.lower() != "y":
    import sys
    sys.exit("exit")

print("## binding the threads")

sch.bind(i_0, "blockIdx.x")
sch.bind(i_2, "threadIdx.x")
print(sch.mod.script())

print("## build the IRModule with cuda backends")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
cuda_a = tvm.nd.array(np.arange(8).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.zeros((8,)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b)
print(cuda_a)
print(cuda_b)
