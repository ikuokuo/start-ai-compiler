# TVMC 调优模型

- [Compiling and Optimizing a Model with TVMC](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html#sphx-glr-tutorial-tvmc-command-line-driver-py)

## TVMC 命令

<!--
conda activate tvm
-->

```bash
pip install scipy

# source .env
alias tvmc="python -m tvm.driver.tvmc"
tvmc -h

# for onnx support
pip install onnx onnxoptimizer
```

## 获取模型

```bash
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx
```

## 编译模型

```bash
tvmc compile --help

tvmc compile \
--target "llvm" \
--output resnet50-v2-7-tvm.tar \
resnet50-v2-7.onnx
```

## 运行模型

```bash
cd frameworks/tvm/tvmc_tune

# for image processing
pip install pillow

# 前处理
python preprocess.py

# 预测
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm.tar

# 后处理
python postprocess.py
```

结果：

```bash
$ python postprocess.py
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```

## 调优模型

```bash
# for tuning search algorithm
pip install xgboost

# for RPC Tracker, ...
pip install tornado cloudpickle psutil

tvmc tune --help

tvmc tune \
--target "llvm" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx

lscpu
# Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
#  Code Name: Products formerly Comet Lake
#  https://ark.intel.com/content/www/us/en/ark/products/201837/intel-core-i710750h-processor-12m-cache-up-to-5-00-ghz.html
```

结果：

```bash
$ tvmc tune \
--target "llvm -mcpu=cometlake" \
--output resnet50-v2-7-autotuner_records.json \
resnet50-v2-7.onnx
...
[Task 25/25]  Current/Best:   11.26/  62.61 GFLOPS | Progress: (40/40) | 30.79 s Done.
```

## 编译调优模型

```bash
tvmc compile \
--target "llvm" \
--tuning-records resnet50-v2-7-autotuner_records.json  \
--output resnet50-v2-7-tvm_autotuned.tar \
resnet50-v2-7.onnx
```

预测：

```bash
tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz \
resnet50-v2-7-tvm_autotuned.tar
```

结果：

```bash
$ python postprocess.py
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356377
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```

## 对比调优前后

```bash
# 调优后
$ tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm_autotuned.tar

Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  57.4865      56.7448      67.7874      55.6124       2.1288

# 调优前
$ tvmc run \
--inputs imagenet_cat.npz \
--output predictions.npz  \
--print-time \
--repeat 100 \
resnet50-v2-7-tvm.tar

Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)
  71.0608      69.8483      83.0061      69.2029       2.9748
```
