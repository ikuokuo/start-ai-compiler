#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name,redefined-outer-name
import tvm
import tvm.testing
from tvm import te
import numpy as np


print("# Writing and Scheduling Vector Addition in TE for CPU")

target = input("  target [llvm]: ")
if not target:
    target = "llvm"

tgt = tvm.target.Target(target=target, host="llvm")


print("\n## Describing the Vector Computation")

# if use constraint, not variable n
# n = tvm.runtime.convert(1024)
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")


print("\n## Create a Default Schedule for the Computation")

# for (int i = 0; i < n; ++i) {
#   C[i] = A[i] + B[i];
# }

s = te.create_schedule(C.op)


print("\n## Compile and Evaluate the Default Schedule")

fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


print("   get a comparison of how fast this version is compared to numpy")

import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))


def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))


log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)


print("\n## Updating the Schedule to Use Paralleism")

s[C].parallel(C.op.axis[0])

# generate the IR of the TE, to apply different schedule operations
print(tvm.lower(s, [A, B, C], simple_mode=True))

fadd_parallel = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")
fadd_parallel(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

evaluate_addition(fadd_parallel, tgt, "parallel", log=log)


print("\n## Updating the Schedule to Use Vectorization")

# Recreate the schedule, since we modified it with the parallel operation in
# the previous example
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# This factor should be chosen to match the number of threads appropriate for
# your CPU. This will vary depending on architecture, but a good rule is
# setting this factor to equal the number of available CPU cores.
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))


print("\n## Comparing the Different Schedules")

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )


print("\n# ------------------------------------------------------------------------------")
s = input("# Targeting Vector Addition for GPUs [Y/n] ")
run_cuda = not s or s.lower() == "y"

if run_cuda:
    # Change this target to the correct backend for you gpu
    print("  target options: cuda (NVIDIA GPUs), rocm (Radeon GPUS), OpenCL (opencl)")
    target = input("  target [cuda]: ")
    if not target:
        target = "cuda"

    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # Recreate the schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    ################################################################################
    # Finally we must bind the iteration axis bx and tx to threads in the GPU
    # compute grid. The naive schedule is not valid for GPUs, and these are
    # specific constructs that allow us to generate code that runs on a GPU.

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    ######################################################################
    # Compilation
    # -----------
    # After we have finished specifying the schedule, we can compile it
    # into a TVM function. By default TVM compiles into a type-erased
    # function that can be directly called from the python side.
    #
    # In the following line, we use tvm.build to create a function.
    # The build function takes the schedule, the desired signature of the
    # function (including the inputs and outputs) as well as target language
    # we want to compile to.
    #
    # The result of compilation fadd is a GPU device function (if GPU is
    # involved) as well as a host wrapper that calls into the GPU
    # function. fadd is the generated host wrapper function, it contains
    # a reference to the generated device function internally.

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ################################################################################
    # The compiled TVM function exposes a concise C API that can be invoked from
    # any language.
    #
    # We provide a minimal array API in python to aid quick testing and prototyping.
    # The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
    #
    # - We first create a GPU device.
    # - Then tvm.nd.array copies the data to the GPU.
    # - ``fadd`` runs the actual computation
    # - ``numpy()`` copies the GPU array back to the CPU (so we can verify correctness).
    #
    # Note that copying the data to and from the memory on the GPU is a required step.

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    ################################################################################
    # Inspect the Generated GPU Code
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # You can inspect the generated code in TVM. The result of tvm.build is a TVM
    # Module. fadd is the host module that contains the host wrapper, it also
    # contains a device module for the CUDA (GPU) function.
    #
    # The following code fetches the device module and prints the content code.

    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd.get_source())

    tgt = tgt_gpu


print("\n# ------------------------------------------------------------------------------")
print("# Saving and Loading Compiled Modules")

from tvm.contrib import cc
from tvm.contrib import utils

temp = utils.tempdir()
print(f"  tempdir={temp.path}")
print(f"  tgt.kind.name={tgt.kind.name}")
fadd.save(temp.relpath("myadd.o"))
if tgt.kind.name == "cuda":
    # fadd.imported_modules[0].save(temp.relpath("myadd.ptx"))
    fadd.imported_modules[0].save(temp.relpath("myadd.cubin"))
if tgt.kind.name == "rocm":
    fadd.imported_modules[0].save(temp.relpath("myadd.hsaco"))
if tgt.kind.name.startswith("opencl"):
    fadd.imported_modules[0].save(temp.relpath("myadd.cl"))
cc.create_shared(temp.relpath("myadd.so"), [temp.relpath("myadd.o")])
print(temp.listdir())


print("## Load Compiled Module")

fadd1 = tvm.runtime.load_module(temp.relpath("myadd.so"))
if tgt.kind.name == "cuda":
    # fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.ptx"))
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cubin"))
    fadd1.import_module(fadd1_dev)
if tgt.kind.name == "rocm":
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.hsaco"))
    fadd1.import_module(fadd1_dev)
if tgt.kind.name.startswith("opencl"):
    fadd1_dev = tvm.runtime.load_module(temp.relpath("myadd.cl"))
    fadd1.import_module(fadd1_dev)

fadd1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


print("## Pack Everything into One Library")
s = input("   currently we support packing of Metal, OpenCL and CUDA modules [Y/n] ")
run_pack = not s or s.lower() == "y"

if run_pack:
    fadd.export_library(temp.relpath("myadd_pack.so"))
    fadd2 = tvm.runtime.load_module(temp.relpath("myadd_pack.so"))
    fadd2(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


print("\n# ------------------------------------------------------------------------------")
tgt_opencl_tag = "Y" if tgt.kind.name.startswith("opencl") else "N"
print(f"# Generate OpenCL Code (target=opencl) [{tgt_opencl_tag}]")

if tgt.kind.name.startswith("opencl"):
    fadd_cl = tvm.build(s, [A, B, C], tgt, name="myadd")
    print("------opencl code------")
    print(fadd_cl.imported_modules[0].get_source())
    dev = tvm.cl(0)
    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd_cl(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


print("\n# ------------------------------------------------------------------------------")
print("TE Scheduling Primitives")
print("  split: splits a specified axis into two axises by the defined factor.")
print("  tile: tiles will split a computation across two axes by the defined factors.")
print("  fuse: fuses two consecutive axises of one computation.")
print("  reorder: can reorder the axises of a computation into a defined order.")
print("  bind: can bind a computation to a specific thread, useful in GPU programming.")
print("  compute_at: by default, TVM will compute tensors at the outermost level of the function, or the root, by default. compute_at specifies that one tensor should be computed at the first axis of computation for another operator.")
print("  compute_inline: when marked inline, a computation will be expanded then inserted into the address where the tensor is required.")
print("  compute_root: moves a computation to the outermost layer, or root, of the function. This means that stage of the computation will be fully computed before it moves on to the next stage.")
print("\nSchedule Primitives")
print("  https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html#schedule-primitives")
