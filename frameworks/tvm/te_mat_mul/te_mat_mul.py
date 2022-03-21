#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import timeit

import tvm
import tvm.testing
from tvm import te
import numpy


print("# Manually Optimizing Matrix Multiplication with TE")


print("\n## Preparation and Performance Baseline")

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

# You will want to adjust the target to match any CPU vector extensions you
# might have. For example, if you're using using Intel AVX2 (Advanced Vector
# Extensions) ISA for SIMD, you can get the best performance by changing the
# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.
# Recall that you're using llvm, you can get this information from the command
# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``
# for additional extensions that your processor might support.

target = input("   target [llvm]: ")
if not target:
    target = "llvm"
target = tvm.target.Target(target=target, host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))


answer = numpy.dot(a.numpy(), b.numpy())
# TVM Matrix Multiplication using TE
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)


def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))


log = []

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 1: Blocking")

bn = 32

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="blocking", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 2: Vectorization")

# Apply the vectorization optimization
s[C].vectorize(yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="vectorization", log=log)

# The generalized IR after vectorization
print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 3: Loop Permutation")

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="loop permutation", log=log
)

# Again, print the new generalized IR
print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 4: Array Packing")

# We have to re-write the algorithm slightly.
packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

# Here is the generated IR after array packing.
print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 5: Optimizing Block Writing Through Caching")

s = te.create_schedule(C.op)

# Allocate write cache
CC = s.cache_write(C, "global")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at yo
s[CC].compute_at(s[C], yo)

# New inner axes
xc, yc = s[CC].op.axis

(k,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="block caching", log=log)

# Here is the generated IR after write cache blocking.
print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Optimization 6: Parallelization")

# parallel
s[C].parallel(xo)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="parallelization", log=log
)

# Here is the generated IR after parallelization.
print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Summary of Matrix Multiplication Example")

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )
