#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring,invalid-name
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing


print("# Define Neural Network in Relay")

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))


print("\n# Compilation (target=CUDA)")

opt_level = 3
target = tvm.target.cuda()
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)


print("\n# Run the generate library")

# create random input
dev = tvm.cuda()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input("data", data)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print(out.flatten()[0:10])


print("\n# Save and Load Compiled Module")

# save the graph, lib and params into separate files
from tvm.contrib import utils

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
print(temp.listdir())


# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)
