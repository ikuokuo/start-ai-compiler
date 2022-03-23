# Relay 开始

- [Quick Start Tutorial for Compiling Deep Learning Models](https://tvm.apache.org/docs/tutorial/relay_quick_start.html#sphx-glr-tutorial-relay-quick-start-py)

代码：

- [relay_start.py](../../frameworks/tvm/relay_start/relay_start.py)

结果：

```bash
$ cd frameworks/tvm/relay_start/
$ python relay_start.py
# Define Neural Network in Relay
#[version = "0.0.5"]
def @main(%data: Tensor[(1, 3, 224, 224), float32], %bn_data_gamma: Tensor[(3), float32], %bn_data_beta: Tensor[(3), float32], %bn_data_moving_mean: Tensor[(3), float32], %bn_data_moving_var: Tensor[(3), float32], %conv0_weight: Tensor[(64, 3, 7, 7), float32], %bn0_gamma: Tensor[(64), float32], %bn0_beta: Tensor[(64), float32], %bn0_moving_mean: Tensor[(64), float32], %bn0_moving_var: Tensor[(64), float32], %stage1_unit1_bn1_gamma: Tensor[(64), float32], %stage1_unit1_bn1_beta: Tensor[(64), float32], %stage1_unit1_bn1_moving_mean: Tensor[(64), float32], %stage1_unit1_bn1_moving_var: Tensor[(64), float32], %stage1_unit1_conv1_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit1_bn2_gamma: Tensor[(64), float32], %stage1_unit1_bn2_beta: Tensor[(64), float32], %stage1_unit1_bn2_moving_mean: Tensor[(64), float32], %stage1_unit1_bn2_moving_var: Tensor[(64), float32], %stage1_unit1_conv2_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit1_sc_weight: Tensor[(64, 64, 1, 1), float32], %stage1_unit2_bn1_gamma: Tensor[(64), float32], %stage1_unit2_bn1_beta: Tensor[(64), float32], %stage1_unit2_bn1_moving_mean: Tensor[(64), float32], %stage1_unit2_bn1_moving_var: Tensor[(64), float32], %stage1_unit2_conv1_weight: Tensor[(64, 64, 3, 3), float32], %stage1_unit2_bn2_gamma: Tensor[(64), float32], %stage1_unit2_bn2_beta: Tensor[(64), float32], %stage1_unit2_bn2_moving_mean: Tensor[(64), float32], %stage1_unit2_bn2_moving_var: Tensor[(64), float32], %stage1_unit2_conv2_weight: Tensor[(64, 64, 3, 3), float32], %stage2_unit1_bn1_gamma: Tensor[(64), float32], %stage2_unit1_bn1_beta: Tensor[(64), float32], %stage2_unit1_bn1_moving_mean: Tensor[(64), float32], %stage2_unit1_bn1_moving_var: Tensor[(64), float32], %stage2_unit1_conv1_weight: Tensor[(128, 64, 3, 3), float32], %stage2_unit1_bn2_gamma: Tensor[(128), float32], %stage2_unit1_bn2_beta: Tensor[(128), float32], %stage2_unit1_bn2_moving_mean: Tensor[(128), float32], %stage2_unit1_bn2_moving_var: Tensor[(128), float32], %stage2_unit1_conv2_weight: Tensor[(128, 128, 3, 3), float32], %stage2_unit1_sc_weight: Tensor[(128, 64, 1, 1), float32], %stage2_unit2_bn1_gamma: Tensor[(128), float32], %stage2_unit2_bn1_beta: Tensor[(128), float32], %stage2_unit2_bn1_moving_mean: Tensor[(128), float32], %stage2_unit2_bn1_moving_var: Tensor[(128), float32], %stage2_unit2_conv1_weight: Tensor[(128, 128, 3, 3), float32], %stage2_unit2_bn2_gamma: Tensor[(128), float32], %stage2_unit2_bn2_beta: Tensor[(128), float32], %stage2_unit2_bn2_moving_mean: Tensor[(128), float32], %stage2_unit2_bn2_moving_var: Tensor[(128), float32], %stage2_unit2_conv2_weight: Tensor[(128, 128, 3, 3), float32], %stage3_unit1_bn1_gamma: Tensor[(128), float32], %stage3_unit1_bn1_beta: Tensor[(128), float32], %stage3_unit1_bn1_moving_mean: Tensor[(128), float32], %stage3_unit1_bn1_moving_var: Tensor[(128), float32], %stage3_unit1_conv1_weight: Tensor[(256, 128, 3, 3), float32], %stage3_unit1_bn2_gamma: Tensor[(256), float32], %stage3_unit1_bn2_beta: Tensor[(256), float32], %stage3_unit1_bn2_moving_mean: Tensor[(256), float32], %stage3_unit1_bn2_moving_var: Tensor[(256), float32], %stage3_unit1_conv2_weight: Tensor[(256, 256, 3, 3), float32], %stage3_unit1_sc_weight: Tensor[(256, 128, 1, 1), float32], %stage3_unit2_bn1_gamma: Tensor[(256), float32], %stage3_unit2_bn1_beta: Tensor[(256), float32], %stage3_unit2_bn1_moving_mean: Tensor[(256), float32], %stage3_unit2_bn1_moving_var: Tensor[(256), float32], %stage3_unit2_conv1_weight: Tensor[(256, 256, 3, 3), float32], %stage3_unit2_bn2_gamma: Tensor[(256), float32], %stage3_unit2_bn2_beta: Tensor[(256), float32], %stage3_unit2_bn2_moving_mean: Tensor[(256), float32], %stage3_unit2_bn2_moving_var: Tensor[(256), float32], %stage3_unit2_conv2_weight: Tensor[(256, 256, 3, 3), float32], %stage4_unit1_bn1_gamma: Tensor[(256), float32], %stage4_unit1_bn1_beta: Tensor[(256), float32], %stage4_unit1_bn1_moving_mean: Tensor[(256), float32], %stage4_unit1_bn1_moving_var: Tensor[(256), float32], %stage4_unit1_conv1_weight: Tensor[(512, 256, 3, 3), float32], %stage4_unit1_bn2_gamma: Tensor[(512), float32], %stage4_unit1_bn2_beta: Tensor[(512), float32], %stage4_unit1_bn2_moving_mean: Tensor[(512), float32], %stage4_unit1_bn2_moving_var: Tensor[(512), float32], %stage4_unit1_conv2_weight: Tensor[(512, 512, 3, 3), float32], %stage4_unit1_sc_weight: Tensor[(512, 256, 1, 1), float32], %stage4_unit2_bn1_gamma: Tensor[(512), float32], %stage4_unit2_bn1_beta: Tensor[(512), float32], %stage4_unit2_bn1_moving_mean: Tensor[(512), float32], %stage4_unit2_bn1_moving_var: Tensor[(512), float32], %stage4_unit2_conv1_weight: Tensor[(512, 512, 3, 3), float32], %stage4_unit2_bn2_gamma: Tensor[(512), float32], %stage4_unit2_bn2_beta: Tensor[(512), float32], %stage4_unit2_bn2_moving_mean: Tensor[(512), float32], %stage4_unit2_bn2_moving_var: Tensor[(512), float32], %stage4_unit2_conv2_weight: Tensor[(512, 512, 3, 3), float32], %bn1_gamma: Tensor[(512), float32], %bn1_beta: Tensor[(512), float32], %bn1_moving_mean: Tensor[(512), float32], %bn1_moving_var: Tensor[(512), float32], %fc1_weight: Tensor[(1000, 512), float32], %fc1_bias: Tensor[(1000), float32]) -> Tensor[(1, 1000), float32] {
  %0 = nn.batch_norm(%data, %bn_data_gamma, %bn_data_beta, %bn_data_moving_mean, %bn_data_moving_var, epsilon=2e-05f, scale=False) /* ty=(Tensor[(1, 3, 224, 224), float32], Tensor[(3), float32], Tensor[(3), float32]) */;
  %1 = %0.0;
  %2 = nn.conv2d(%1, %conv0_weight, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %3 = nn.batch_norm(%2, %bn0_gamma, %bn0_beta, %bn0_moving_mean, %bn0_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 112, 112), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %4 = %3.0;
  %5 = nn.relu(%4) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %6 = nn.max_pool2d(%5, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %7 = nn.batch_norm(%6, %stage1_unit1_bn1_gamma, %stage1_unit1_bn1_beta, %stage1_unit1_bn1_moving_mean, %stage1_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %8 = %7.0;
  %9 = nn.relu(%8) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %10 = nn.conv2d(%9, %stage1_unit1_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %11 = nn.batch_norm(%10, %stage1_unit1_bn2_gamma, %stage1_unit1_bn2_beta, %stage1_unit1_bn2_moving_mean, %stage1_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %12 = %11.0;
  %13 = nn.relu(%12) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %14 = nn.conv2d(%13, %stage1_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %15 = nn.conv2d(%9, %stage1_unit1_sc_weight, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %16 = add(%14, %15) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %17 = nn.batch_norm(%16, %stage1_unit2_bn1_gamma, %stage1_unit2_bn1_beta, %stage1_unit2_bn1_moving_mean, %stage1_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %18 = %17.0;
  %19 = nn.relu(%18) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %20 = nn.conv2d(%19, %stage1_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %21 = nn.batch_norm(%20, %stage1_unit2_bn2_gamma, %stage1_unit2_bn2_beta, %stage1_unit2_bn2_moving_mean, %stage1_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %22 = %21.0;
  %23 = nn.relu(%22) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %24 = nn.conv2d(%23, %stage1_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %25 = add(%24, %16) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %26 = nn.batch_norm(%25, %stage2_unit1_bn1_gamma, %stage2_unit1_bn1_beta, %stage2_unit1_bn1_moving_mean, %stage2_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 64, 56, 56), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
  %27 = %26.0;
  %28 = nn.relu(%27) /* ty=Tensor[(1, 64, 56, 56), float32] */;
  %29 = nn.conv2d(%28, %stage2_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %30 = nn.batch_norm(%29, %stage2_unit1_bn2_gamma, %stage2_unit1_bn2_beta, %stage2_unit1_bn2_moving_mean, %stage2_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %31 = %30.0;
  %32 = nn.relu(%31) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %33 = nn.conv2d(%32, %stage2_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %34 = nn.conv2d(%28, %stage2_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %35 = add(%33, %34) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %36 = nn.batch_norm(%35, %stage2_unit2_bn1_gamma, %stage2_unit2_bn1_beta, %stage2_unit2_bn1_moving_mean, %stage2_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %37 = %36.0;
  %38 = nn.relu(%37) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %39 = nn.conv2d(%38, %stage2_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %40 = nn.batch_norm(%39, %stage2_unit2_bn2_gamma, %stage2_unit2_bn2_beta, %stage2_unit2_bn2_moving_mean, %stage2_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %41 = %40.0;
  %42 = nn.relu(%41) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %43 = nn.conv2d(%42, %stage2_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %44 = add(%43, %35) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %45 = nn.batch_norm(%44, %stage3_unit1_bn1_gamma, %stage3_unit1_bn1_beta, %stage3_unit1_bn1_moving_mean, %stage3_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 128, 28, 28), float32], Tensor[(128), float32], Tensor[(128), float32]) */;
  %46 = %45.0;
  %47 = nn.relu(%46) /* ty=Tensor[(1, 128, 28, 28), float32] */;
  %48 = nn.conv2d(%47, %stage3_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %49 = nn.batch_norm(%48, %stage3_unit1_bn2_gamma, %stage3_unit1_bn2_beta, %stage3_unit1_bn2_moving_mean, %stage3_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %50 = %49.0;
  %51 = nn.relu(%50) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %52 = nn.conv2d(%51, %stage3_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %53 = nn.conv2d(%47, %stage3_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %54 = add(%52, %53) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %55 = nn.batch_norm(%54, %stage3_unit2_bn1_gamma, %stage3_unit2_bn1_beta, %stage3_unit2_bn1_moving_mean, %stage3_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %56 = %55.0;
  %57 = nn.relu(%56) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %58 = nn.conv2d(%57, %stage3_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %59 = nn.batch_norm(%58, %stage3_unit2_bn2_gamma, %stage3_unit2_bn2_beta, %stage3_unit2_bn2_moving_mean, %stage3_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %60 = %59.0;
  %61 = nn.relu(%60) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %62 = nn.conv2d(%61, %stage3_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %63 = add(%62, %54) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %64 = nn.batch_norm(%63, %stage4_unit1_bn1_gamma, %stage4_unit1_bn1_beta, %stage4_unit1_bn1_moving_mean, %stage4_unit1_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 256, 14, 14), float32], Tensor[(256), float32], Tensor[(256), float32]) */;
  %65 = %64.0;
  %66 = nn.relu(%65) /* ty=Tensor[(1, 256, 14, 14), float32] */;
  %67 = nn.conv2d(%66, %stage4_unit1_conv1_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %68 = nn.batch_norm(%67, %stage4_unit1_bn2_gamma, %stage4_unit1_bn2_beta, %stage4_unit1_bn2_moving_mean, %stage4_unit1_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %69 = %68.0;
  %70 = nn.relu(%69) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %71 = nn.conv2d(%70, %stage4_unit1_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %72 = nn.conv2d(%66, %stage4_unit1_sc_weight, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %73 = add(%71, %72) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %74 = nn.batch_norm(%73, %stage4_unit2_bn1_gamma, %stage4_unit2_bn1_beta, %stage4_unit2_bn1_moving_mean, %stage4_unit2_bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %75 = %74.0;
  %76 = nn.relu(%75) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %77 = nn.conv2d(%76, %stage4_unit2_conv1_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %78 = nn.batch_norm(%77, %stage4_unit2_bn2_gamma, %stage4_unit2_bn2_beta, %stage4_unit2_bn2_moving_mean, %stage4_unit2_bn2_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %79 = %78.0;
  %80 = nn.relu(%79) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %81 = nn.conv2d(%80, %stage4_unit2_conv2_weight, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %82 = add(%81, %73) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %83 = nn.batch_norm(%82, %bn1_gamma, %bn1_beta, %bn1_moving_mean, %bn1_moving_var, epsilon=2e-05f) /* ty=(Tensor[(1, 512, 7, 7), float32], Tensor[(512), float32], Tensor[(512), float32]) */;
  %84 = %83.0;
  %85 = nn.relu(%84) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %86 = nn.global_avg_pool2d(%85) /* ty=Tensor[(1, 512, 1, 1), float32] */;
  %87 = nn.batch_flatten(%86) /* ty=Tensor[(1, 512), float32] */;
  %88 = nn.dense(%87, %fc1_weight, units=1000) /* ty=Tensor[(1, 1000), float32] */;
  %89 = nn.bias_add(%88, %fc1_bias, axis=-1) /* ty=Tensor[(1, 1000), float32] */;
  nn.softmax(%89) /* ty=Tensor[(1, 1000), float32] */
}


# Compilation (target=CUDA)
/home/john/Codes/Star/tvm/python/tvm/target/target.py:282: UserWarning: Try specifying cuda arch by adding 'arch=sm_xx' to your target.
  warnings.warn("Try specifying cuda arch by adding 'arch=sm_xx' to your target.")
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.

# Run the generate library
[0.00089283 0.00103331 0.0009094  0.00102275 0.00108751 0.00106737
 0.00106262 0.00095838 0.00110792 0.00113151]

# Save and Load Compiled Module
['deploy_lib.tar']
[0.00089283 0.00103331 0.0009094  0.00102275 0.00108751 0.00106737
 0.00106262 0.00095838 0.00110792 0.00113151]
```
