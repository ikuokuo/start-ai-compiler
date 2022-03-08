# AutoTVM 调优模型

- [Compiling and Optimizing a Model with the Python Interface (AutoTVM)](https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html#sphx-glr-tutorial-autotvm-relay-x86-py)

代码：

- [autotvm_tune.py](../../frameworks/tvm/autotvm_tune/autotvm_tune.py)

结果：

```bash
$ time python autotvm_tune.py
# TVM 编译运行模型
## Downloading and Loading the ONNX Model
## Downloading, Preprocessing, and Loading the Test Image
## Compile the Model With Relay
target [llvm]: llvm -mcpu=core-avx2
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
## Execute on the TVM Runtime
## Collect Basic Performance Data
{'mean': 44.97057118016528, 'median': 42.52320024970686, 'std': 6.870915251002107}
## Postprocess the output
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
# AutoTVM 调优模型 [Y/n]
## Tune the model
[Task  1/25]  Current/Best:  156.96/ 353.76 GFLOPS | Progress: (10/10) | 4.78 s Done.
[Task  2/25]  Current/Best:   54.66/ 241.25 GFLOPS | Progress: (10/10) | 2.88 s Done.
[Task  3/25]  Current/Best:  116.71/ 241.30 GFLOPS | Progress: (10/10) | 3.48 s Done.
[Task  4/25]  Current/Best:  119.92/ 184.18 GFLOPS | Progress: (10/10) | 3.48 s Done.
[Task  5/25]  Current/Best:   48.92/ 158.38 GFLOPS | Progress: (10/10) | 3.13 s Done.
[Task  6/25]  Current/Best:  156.89/ 230.95 GFLOPS | Progress: (10/10) | 2.82 s Done.
[Task  7/25]  Current/Best:   92.33/ 241.99 GFLOPS | Progress: (10/10) | 2.40 s Done.
[Task  8/25]  Current/Best:   50.04/ 331.82 GFLOPS | Progress: (10/10) | 2.64 s Done.
[Task  9/25]  Current/Best:  188.47/ 409.93 GFLOPS | Progress: (10/10) | 4.44 s Done.
[Task 10/25]  Current/Best:   44.81/ 181.67 GFLOPS | Progress: (10/10) | 2.32 s Done.
[Task 11/25]  Current/Best:   83.74/ 312.66 GFLOPS | Progress: (10/10) | 2.74 s Done.
[Task 12/25]  Current/Best:   96.48/ 294.40 GFLOPS | Progress: (10/10) | 2.82 s Done.
[Task 13/25]  Current/Best:  123.74/ 354.34 GFLOPS | Progress: (10/10) | 2.62 s Done.
[Task 14/25]  Current/Best:   23.76/ 178.71 GFLOPS | Progress: (10/10) | 2.90 s Done.
[Task 15/25]  Current/Best:  119.18/ 534.63 GFLOPS | Progress: (10/10) | 2.49 s Done.
[Task 16/25]  Current/Best:  101.24/ 172.92 GFLOPS | Progress: (10/10) | 2.49 s Done.
[Task 17/25]  Current/Best:  309.85/ 309.85 GFLOPS | Progress: (10/10) | 2.69 s Done.
[Task 18/25]  Current/Best:   54.45/ 368.31 GFLOPS | Progress: (10/10) | 2.46 s Done.
[Task 19/25]  Current/Best:   78.69/ 162.43 GFLOPS | Progress: (10/10) | 3.29 s Done.
[Task 20/25]  Current/Best:   40.78/ 317.50 GFLOPS | Progress: (10/10) | 4.52 s Done.
[Task 21/25]  Current/Best:  169.03/ 296.36 GFLOPS | Progress: (10/10) | 3.95 s Done.
[Task 22/25]  Current/Best:   90.96/ 210.43 GFLOPS | Progress: (10/10) | 2.28 s Done.
[Task 23/25]  Current/Best:   48.93/ 217.36 GFLOPS | Progress: (10/10) | 2.87 s Done.
[Task 25/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/10) | 0.00 s Done.
[Task 25/25]  Current/Best:   25.50/  33.86 GFLOPS | Progress: (10/10) | 9.28 s Done.
## Compiling an Optimized Model with Tuning Data
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
## Comparing the Tuned and Untuned Models
optimized: {'mean': 34.736288779822644, 'median': 34.547542000655085, 'std': 0.5144378649382363}
unoptimized: {'mean': 44.97057118016528, 'median': 42.52320024970686, 'std': 6.870915251002107}

real    3m23.904s
user    5m2.900s
sys     5m37.099s
```
