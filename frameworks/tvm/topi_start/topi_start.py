#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np


print("# Basic example")

print("## compute the sum of rows")
# B = numpy.sum(A, axis=1)

n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)

print(tvm.lower(s, [A], simple_mode=True))


print("## replace te.reduce_axis(), te.compute() with topi.sum()")

C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
print(tvm.lower(ts, [A], simple_mode=True))


print("\n# Numpy-style operator overloading")

print("## topi.broadcast_add() using '+'")

x, y = 100, 10
a = te.placeholder((x, y, y), name="a")
b = te.placeholder((y, y), name="b")
c = a + b  # same as topi.broadcast_add
d = a * b  # same as topi.broadcast_mul


print("\n# Generic schedules and fusing operations (CUDA)")

e = topi.elemwise_sum([c, d])
f = e / 2.0
g = topi.sum(f)
with tvm.target.cuda():
    sg = topi.cuda.schedule_reduce(g)
    print(tvm.lower(sg, [a, b], simple_mode=True))

print(sg.stages)

print("## test the correctness")
func = tvm.build(sg, [a, b, g], "cuda")
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)
a_nd = tvm.nd.array(a_np, dev)
b_nd = tvm.nd.array(b_np, dev)
g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), dev)
func(a_nd, b_nd, g_nd)
tvm.testing.assert_allclose(g_nd.numpy(), g_np, rtol=1e-5)

print("## schedule softmax")

tarray = te.placeholder((512, 512), name="tarray")
softmax_topi = topi.nn.softmax(tarray)
with tvm.target.Target("cuda"):
    sst = topi.cuda.schedule_softmax(softmax_topi)
    print(tvm.lower(sst, [tarray], simple_mode=True))


print("\n# Fusing convolutions (CUDA)")

print("## fuse topi.nn.conv2d and topi.nn.relu together")

data = te.placeholder((1, 3, 224, 224))
kernel = te.placeholder((10, 3, 5, 5))

with tvm.target.Target("cuda"):
    conv = topi.cuda.conv2d_nchw(data, kernel, 1, 2, 1)
    out = topi.nn.relu(conv)
    sconv = topi.cuda.schedule_conv2d_nchw([out])
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))
