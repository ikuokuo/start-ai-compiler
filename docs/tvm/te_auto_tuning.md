# TE 自动调优

- [Optimizing Operators with Schedule Templates and AutoTVM](https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html#sphx-glr-tutorial-autotvm-matmul-x86-py)

代码：

- [te_auto_tuning.py](../../frameworks/tvm/te_auto_tuning/te_auto_tuning.py)

结果：

```bash
$ cd frameworks/tvm/te_auto_tuning/
$ python te_auto_tuning.py
# Basic Matrix Multiplication with TE

# Matrix Multiplication with AutoTVM

# A Basic Matrix Multiplication Template

# A Matrix Multiplication Template with the Advanced Parameter API

# Use AutoTVM to Optimize the Matrix Multiplication

## Auto-tuners in TVM
Some of the tuner strategies provided by TVM include:
  tvm.autotvm.tuner.RandomTuner: Enumerate the space in a random order
  tvm.autotvm.tuner.GridSearchTuner: Enumerate the space in a grid search order
  tvm.autotvm.tuner.GATuner: Using genetic algorithm to search through the space
  tvm.autotvm.tuner.XGBTuner: Uses a model based method. Train a XGBoost model to predict the speed of lowered IR and pick the next batch according to the prediction.

## Begin tuning
ConfigSpace (len=100, space_map=
   0 tile_y: Split(policy=factors, product=512, num_outputs=2) len=10
   1 tile_x: Split(policy=factors, product=512, num_outputs=2) len=10
)
waiting for device...
device available
Get devices for measurement successfully!
No: 1   GFLOPS: 2.10/2.10       result: MeasureResult(costs=(0.1277072908,), error_no=MeasureErrorNo.NO_ERROR, all_cost=2.139360189437866, timestamp=1647844396.9014063)  [('tile_y', [-1, 256]), ('tile_x', [-1, 4])],None,28
No: 2   GFLOPS: 1.17/2.10       result: MeasureResult(costs=(0.2292009042,), error_no=MeasureErrorNo.NO_ERROR, all_cost=3.7636666297912598, timestamp=1647844400.6962876) [('tile_y', [-1, 128]), ('tile_x', [-1, 2])],None,17
No: 3   GFLOPS: 7.53/7.53       result: MeasureResult(costs=(0.03566494839999999,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.7075166702270508, timestamp=1647844401.4487107)  [('tile_y', [-1, 4]), ('tile_x', [-1, 2])],None,12
No: 4   GFLOPS: 15.32/15.32     result: MeasureResult(costs=(0.0175250908,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.3513762950897217, timestamp=1647844401.8682544) [('tile_y', [-1, 1]), ('tile_x', [-1, 512])],None,90
No: 5   GFLOPS: 19.52/19.52     result: MeasureResult(costs=(0.013749547999999997,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.31203246116638184, timestamp=1647844402.2267017)        [('tile_y', [-1, 4]), ('tile_x', [-1, 512])],None,92
No: 6   GFLOPS: 14.44/19.52     result: MeasureResult(costs=(0.018594258,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.4802713394165039, timestamp=1647844402.7050195)  [('tile_y', [-1, 8]), ('tile_x', [-1, 64])],None,63
No: 7   GFLOPS: 13.38/19.52     result: MeasureResult(costs=(0.020057111199999998,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.42830610275268555, timestamp=1647844403.1675782)        [('tile_y', [-1, 256]), ('tile_x', [-1, 256])],None,88
No: 8   GFLOPS: 18.25/19.52     result: MeasureResult(costs=(0.014710455,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.48306918144226074, timestamp=1647844403.5425029) [('tile_y', [-1, 64]), ('tile_x', [-1, 64])],None,66
No: 9   GFLOPS: 5.97/19.52      result: MeasureResult(costs=(0.0449551786,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.8349404335021973, timestamp=1647844404.4447997) [('tile_y', [-1, 1]), ('tile_x', [-1, 4])],None,20
No: 10  GFLOPS: 10.64/19.52     result: MeasureResult(costs=(0.0252177376,), error_no=MeasureErrorNo.NO_ERROR, all_cost=0.5461795330047607, timestamp=1647844404.994233)  [('tile_y', [-1, 64]), ('tile_x', [-1, 16])],None,46
Finish loading 10 records
```
